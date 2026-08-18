#!/usr/bin/env bash
set -euo pipefail

exec /usr/sbin/sshd -D -e
