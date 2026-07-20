#!/bin/bash
echo "----------------------------------------------"
echo "Running Phala Cloud Pre-Launch Script v0.0.15"
echo "----------------------------------------------"
set -e

# Function: notify host

notify_host() {
    if command -v dstack-util >/dev/null 2>&1; then
        dstack-util notify-host -e "$1" -d "$2"
    else
        tdxctl notify-host -e "$1" -d "$2"
    fi
}

notify_host_hoot_info() {
    notify_host "boot.progress" "$1"
}

notify_host_hoot_error() {
    notify_host "boot.error" "$1"
}

# Function: Perform Docker cleanup
perform_cleanup() {
    echo "Pruning unused images"
    docker image prune -af
    echo "Pruning unused volumes"
    docker volume prune -f
    notify_host_hoot_info "docker cleanup completed"
}

# Function: Check Docker login status without exposing credentials
check_docker_login() {
    local registry="$1"

    # When registry is specified, check auth entry for that registry in Docker config
    if [[ -n "$registry" ]]; then
        local docker_config_path="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
        if [[ -f "$docker_config_path" ]] && grep -q "$registry" "$docker_config_path"; then
            return 0
        else
            return 1
        fi
    fi

    # Fallback check when no explicit registry is provided
    if docker info 2>/dev/null | grep -q "Username"; then
        return 0
    else
        return 1
    fi
}

# Main logic starts here
echo "Starting login process..."

# Check if Docker credentials exist
if [[ -n "$DSTACK_DOCKER_USERNAME" && -n "$DSTACK_DOCKER_PASSWORD" ]]; then
    echo "Docker credentials found"
    DOCKER_REGISTRY_TARGET="${DSTACK_DOCKER_REGISTRY:-docker.io}"
    echo "Target Docker registry: $DOCKER_REGISTRY_TARGET"

    # Check if already logged in
    if check_docker_login "$DSTACK_DOCKER_REGISTRY"; then
        echo "Already logged in to Docker registry: $DOCKER_REGISTRY_TARGET"
    else
        echo "Logging in to Docker registry: $DOCKER_REGISTRY_TARGET"
        # Login without exposing password in process list
        if [[ -n "$DSTACK_DOCKER_REGISTRY" ]]; then
            echo "$DSTACK_DOCKER_PASSWORD" | docker login -u "$DSTACK_DOCKER_USERNAME" --password-stdin "$DSTACK_DOCKER_REGISTRY"
        else
            echo "$DSTACK_DOCKER_PASSWORD" | docker login -u "$DSTACK_DOCKER_USERNAME" --password-stdin
        fi

        if [ $? -eq 0 ]; then
            echo "Docker login successful: $DOCKER_REGISTRY_TARGET"
        else
            echo "Docker login failed: $DOCKER_REGISTRY_TARGET"
            notify_host_hoot_error "docker login failed"
            exit 1
        fi
    fi
# Check if AWS ECR credentials exist
elif [[ -n "$DSTACK_AWS_ACCESS_KEY_ID" && -n "$DSTACK_AWS_SECRET_ACCESS_KEY" && -n "$DSTACK_AWS_REGION" && -n "$DSTACK_AWS_ECR_REGISTRY" ]]; then
    echo "AWS ECR credentials found"

    # Check if AWS CLI is installed
    if [ ! -f "./aws/dist/aws" ]; then
        notify_host_hoot_info "awscli not installed, installing..."
        echo "AWS CLI not installed, installing..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.24.14.zip" -o "awscliv2.zip"
        echo "6ff031a26df7daebbfa3ccddc9af1450 awscliv2.zip" | md5sum -c
        if [ $? -ne 0 ]; then
            echo "MD5 checksum failed"
            notify_host_hoot_error "awscli install failed"
            exit 1
        fi
        unzip awscliv2.zip &> /dev/null
    else
        echo "AWS CLI is already installed: ./aws/dist/aws"
    fi

    # Set AWS credentials as environment variables
    export AWS_ACCESS_KEY_ID="$DSTACK_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$DSTACK_AWS_SECRET_ACCESS_KEY"
    export AWS_DEFAULT_REGION="$DSTACK_AWS_REGION"

    # Set session token if provided (for temporary credentials)
    if [[ -n "$DSTACK_AWS_SESSION_TOKEN" ]]; then
        echo "AWS session token found, using temporary credentials"
        export AWS_SESSION_TOKEN="$DSTACK_AWS_SESSION_TOKEN"
    fi

    # Test AWS credentials before attempting ECR login
    echo "Testing AWS credentials..."
    if ! ./aws/dist/aws sts get-caller-identity &> /dev/null; then
        echo "AWS credentials test failed"
        # For session token credentials, this might be expected if they're expired
        # Log warning but don't fail startup
        if [[ -n "$DSTACK_AWS_SESSION_TOKEN" ]]; then
            echo "Warning: AWS temporary credentials may have expired, continuing startup"
            notify_host_hoot_info "AWS temporary credentials may have expired"
        else
            echo "AWS credentials test failed"
            notify_host_hoot_error "Invalid AWS credentials"
            exit 1
        fi
    else
        echo "Logging in to AWS ECR..."
        ./aws/dist/aws ecr get-login-password --region $DSTACK_AWS_REGION | docker login --username AWS --password-stdin "$DSTACK_AWS_ECR_REGISTRY"
        if [ $? -eq 0 ]; then
            echo "AWS ECR login successful"
            notify_host_hoot_info "AWS ECR login successful"
        else
            echo "AWS ECR login failed"
            # For session token credentials, don't fail startup if login fails
            if [[ -n "$DSTACK_AWS_SESSION_TOKEN" ]]; then
                echo "Warning: AWS ECR login failed with temporary credentials, continuing startup"
                notify_host_hoot_info "AWS ECR login failed with temporary credentials"
            else
                notify_host_hoot_error "AWS ECR login failed"
                exit 1
            fi
        fi
    fi
fi

perform_cleanup

#
# GHCR image pull access verification (pure HTTP, no docker daemon)
#
if [[ "$DOCKER_REGISTRY_TARGET" == "ghcr.io" && -n "$DSTACK_DOCKER_USERNAME" && -n "$DSTACK_DOCKER_PASSWORD" ]]; then
    COMPOSE_IMAGES=$(grep 'image:' /dstack/docker-compose.yaml 2>/dev/null | awk '{print $2}' | tr -d '"'"'" || true)
    for img in $COMPOSE_IMAGES; do
        [[ "$img" != ghcr.io/* ]] && continue
        repo="${img#ghcr.io/}"; repo="${repo%%:*}"
        tag="${img##*:}"; [[ "$tag" == "$img" || "$tag" == "$repo" ]] && tag="latest"
        echo "Verifying GHCR pull access: $img"
        token=$(curl -sf -u "$DSTACK_DOCKER_USERNAME:$DSTACK_DOCKER_PASSWORD"             "https://ghcr.io/token?service=ghcr.io&scope=repository:${repo}:pull" | jq -r '.token // empty' || true)
        if [[ -z "$token" ]]; then
            echo "ERROR: GHCR token exchange failed for $img"
            notify_host_hoot_error "GHCR token exchange failed: $img"
            exit 1
        fi
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $token"             -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json"             "https://ghcr.io/v2/${repo}/manifests/${tag}")
        if [[ "$http_code" != "200" ]]; then
            echo "ERROR: GHCR pull access denied for $img (HTTP $http_code)"
            notify_host_hoot_error "GHCR pull access denied: $img (HTTP $http_code)"
            exit 1
        fi
        echo "GHCR pull access OK: $img"
    done
fi

#
# Pull latest images from docker-compose.yaml so existing CVMs pick up new tags.
# Pull is fail-soft: short-lived tokens may expire between login and pull, but
# falling back to already-cached images is preferable to blocking boot.
#
echo "Images before pull:"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} ({{.CreatedSince}})'

echo "Pulling latest images from /dstack/docker-compose.yaml..."
if docker compose -f /dstack/docker-compose.yaml pull; then
    echo "docker compose pull completed"
    notify_host_hoot_info "docker compose pull completed"
else
    echo "WARNING: docker compose pull failed; continuing with existing images"
    notify_host_hoot_info "docker compose pull failed; using existing images"
fi

echo "Images after pull:"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} ({{.CreatedSince}})'

#
# Set root password.
#
echo "Setting root password.."

# Check if password files are writable
PASSWD_WRITABLE=true
if [ ! -w /etc/passwd ]; then
    echo "Warning: /etc/passwd is read-only"
    PASSWD_WRITABLE=false
fi
if [ ! -w /etc/shadow ]; then
    echo "Warning: /etc/shadow is read-only"
    PASSWD_WRITABLE=false
fi

if [ "$PASSWD_WRITABLE" = "false" ]; then
    echo "Skipping password setup due to read-only file system"
else
    # Check if chpasswd is available
    if command -v chpasswd >/dev/null 2>&1; then
        echo "Using chpasswd method"

        if [ -n "$DSTACK_ROOT_PASSWORD" ]; then
            echo "Setting root password from user.."
            echo "root:$DSTACK_ROOT_PASSWORD" | chpasswd
            unset DSTACK_ROOT_PASSWORD
            echo "Root password set/updated from DSTACK_ROOT_PASSWORD"
        elif [ -z "$(grep '^root:' /etc/shadow 2>/dev/null | cut -d: -f2)" ]; then
            echo "Setting random root password.."
            DSTACK_ROOT_PASSWORD=$(
                LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | dd bs=1 count=32 2>/dev/null
            )
            echo "root:$DSTACK_ROOT_PASSWORD" | chpasswd
            unset DSTACK_ROOT_PASSWORD
            echo "Root password set (random auto-init)"
        else
            echo "Root password already set; no changes."
        fi
    else
        echo "Using passwd method"

        if [ -n "$DSTACK_ROOT_PASSWORD" ]; then
            echo "Setting root password from user.."
            echo "$DSTACK_ROOT_PASSWORD" | passwd --stdin root 2>/dev/null                 || printf '%s
%s
' "$DSTACK_ROOT_PASSWORD" "$DSTACK_ROOT_PASSWORD" | passwd root
            unset DSTACK_ROOT_PASSWORD
            echo "Root password set/updated from DSTACK_ROOT_PASSWORD"
        elif [ -z "$(grep '^root:' /etc/shadow 2>/dev/null | cut -d: -f2)" ]; then
            echo "Setting random root password.."
            DSTACK_ROOT_PASSWORD=$(
                LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | dd bs=1 count=32 2>/dev/null
            )
            echo "$DSTACK_ROOT_PASSWORD" | passwd --stdin root 2>/dev/null                 || printf '%s
%s
' "$DSTACK_ROOT_PASSWORD" "$DSTACK_ROOT_PASSWORD" | passwd root
            unset DSTACK_ROOT_PASSWORD
            echo "Root password set (random auto-init)"
        else
            echo "Root password already set; no changes."
        fi
    fi
fi

#
# Set SSH authorized keys
#
if mkdir -p /home/root/.ssh 2>/dev/null; then
    if [[ -n "$DSTACK_ROOT_PUBLIC_KEY" ]]; then
        echo "$DSTACK_ROOT_PUBLIC_KEY" > /home/root/.ssh/authorized_keys
        unset $DSTACK_ROOT_PUBLIC_KEY
        echo "Root public key set"
    fi
    if [[ -n "$DSTACK_AUTHORIZED_KEYS" ]]; then
        echo "$DSTACK_AUTHORIZED_KEYS" > /home/root/.ssh/authorized_keys
        unset $DSTACK_AUTHORIZED_KEYS
        echo "Root authorized_keys set"
    fi

    if [[ -f /dstack/user_config ]] && jq empty /dstack/user_config 2>/dev/null; then
        if [[ $(jq 'has("ssh_authorized_keys")' /dstack/user_config 2>/dev/null) == "true" ]]; then
            jq -j '.ssh_authorized_keys' /dstack/user_config >> /home/root/.ssh/authorized_keys
            # Remove duplicates if there are multiple keys
            if [[ $(cat /home/root/.ssh/authorized_keys | wc -l) -gt 1 ]]; then
                sort -u /home/root/.ssh/authorized_keys > /home/root/.ssh/authorized_keys.tmp
                mv /home/root/.ssh/authorized_keys.tmp /home/root/.ssh/authorized_keys
            fi
            echo "Set root authorized_keys from user preferences, total" $(cat /home/root/.ssh/authorized_keys | wc -l) "keys"
        fi
    fi
else
    echo "Warning: Cannot create /home/root/.ssh directory (read-only file system?)"
    echo "Skipping SSH key setup"
fi

if [[ -S /var/run/dstack.sock ]]; then
    export DSTACK_APP_ID=$(curl -s --unix-socket /var/run/dstack.sock http://dstack/Info | jq -j .app_id)
elif [[ -S /var/run/tappd.sock ]]; then
    export DSTACK_APP_ID=$(curl -s --unix-socket /var/run/tappd.sock http://dstack/prpc/Tappd.Info | jq -j .app_id)
fi
# Check if DSTACK_GATEWAY_DOMAIN is not set, try to get it from user_config or app-compose.json
# Priority: user_config > app-compose.json
if [[ -z "$DSTACK_GATEWAY_DOMAIN" ]]; then
    # First try to get from /dstack/user_config if it exists and is valid JSON
    if [[ -f /dstack/user_config ]] && jq empty /dstack/user_config 2>/dev/null; then
        if [[ $(jq 'has("default_gateway_domain")' /dstack/user_config 2>/dev/null) == "true" ]]; then
            export DSTACK_GATEWAY_DOMAIN=$(jq -j '.default_gateway_domain' /dstack/user_config)
        fi
    fi

    # If still not set, try to get from app-compose.json
    if [[ -z "$DSTACK_GATEWAY_DOMAIN" ]] && [[ $(jq 'has("default_gateway_domain")' app-compose.json) == "true" ]]; then
        export DSTACK_GATEWAY_DOMAIN=$(jq -j '.default_gateway_domain' app-compose.json)
    fi
fi
if [[ -n "$DSTACK_GATEWAY_DOMAIN" ]]; then
    export DSTACK_APP_DOMAIN=$DSTACK_APP_ID"."$DSTACK_GATEWAY_DOMAIN
fi

echo "----------------------------------------------"
echo "Script execution completed"
echo "----------------------------------------------"
