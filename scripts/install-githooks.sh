#!/usr/bin/env bash
# Point this clone at repo-managed hooks under .githooks/
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
git -C "$root" config core.hooksPath .githooks
echo "core.hooksPath=.githooks"
ls -la "$root/.githooks"
