# gbase Terraform — droplets + firewall

Provisions **exactly two** DigitalOcean droplets and **one** cloud firewall.

| Resource | Spec |
|----------|------|
| `gbase-staging` | `s-8vcpu-16gb-amd`, `nyc1`, Ubuntu 24.04 |
| `gbase-prod` | same |
| `gbase-hosts` firewall | TCP 22 from operator `/32` only; TCP 80/443 + ICMP world; outbound open |


> **Region note:** Plan text asked for `nyc3`. On this DO account the
> `s-8vcpu-16gb-amd` slug is **not** offered in `nyc3` (API regions list +
> create returns 422). `nyc1` is the New York metro region where the size is
> available. Override `region` in `terraform.tfvars` if capacity returns to nyc3.

> **Size note:** the plan named `s-8vcpu-16gb` ($96). That slug is no longer in
> the DO catalog for this account. `s-8vcpu-16gb-amd` is the current 8 vCPU /
> 16 GB Basic AMD size (~$112/mo). Memory/vCPU match the approved capacity.

## Auth

```bash
export DIGITALOCEAN_TOKEN=...   # never commit; doctl config is fine source
```

## Apply

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# edit ssh_key_ids + operator_ipv4_cidrs
terraform init
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
```

State files are **gitignored**. Prefer a remote backend before multi-operator use.

## Cloud-init (no secrets)

`cloud-init.yaml.tftpl` installs Docker Engine + Compose plugin + `age` only.
DO metadata is world-readable on the droplet path — **no tokens, wallets, or
age private keys** in user-data.

## Age secret delivery (R11)

Secrets never travel through Terraform or cloud-init.

1. **Generate / hold identity off-box** (operator laptop or HSM path):
   - Private: `/root/.gbase-secrets/age-identity.txt` mode `600` (example path)
   - Public: `age-keygen -y age-identity.txt` → recipient string
2. **Encrypt env files on the operator machine:**
   ```bash
   ./deploy/scripts/age-encrypt-env.sh \
     --recipient age1... \
     --src-dir deploy/env \
     --out-dir /tmp/gbase-env-age
   ```
3. **Deliver identity out of band once** (USB, 1Password, `scp` over the
   already-firewalled SSH from the operator IP — not via cloud-init):
   ```bash
   ssh root@<droplet-ip> 'mkdir -p /etc/gbase && chmod 700 /etc/gbase'
   scp /path/to/age-identity.txt root@<droplet-ip>:/etc/gbase/age-identity.txt
   ssh root@<droplet-ip> 'chmod 600 /etc/gbase/age-identity.txt'
   ```
4. **Push ciphertext and materialize on the box:**
   ```bash
   ./deploy/scripts/age-push-env.sh --host root@<droplet-ip> --age-dir /tmp/gbase-env-age
   # on droplet:
   export AGE_IDENTITY=/etc/gbase/age-identity.txt
   /opt/gbase/deploy/scripts/materialize-env.sh
   ```

Ciphertext (`*.env.age`) may live in a private ops store; plaintext `*.env`
stays mode `0600` on the droplet only. Git already ignores `*.age` and
`deploy/env/*.env`.

## Destroy

```bash
terraform destroy
```

Does not touch unrelated DO resources (k8s workers, other firewalls).
