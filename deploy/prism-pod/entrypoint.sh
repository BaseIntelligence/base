#!/usr/bin/env bash
set -euo pipefail

install -d -m 0700 /root/.ssh
mkdir -p /run/sshd
if (( $# > 0 )); then
  printf '%s\n' "$*" > /root/.ssh/authorized_keys
  chmod 0600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
touch /root/container_ready
exec /usr/sbin/sshd -D -e
