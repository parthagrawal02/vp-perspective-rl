#!/bin/bash
# Clone NVlabs/DiffusionNFT at the exact upstream commit we used (cbb14f8) into $VP_NFT_DIR and apply our patch.
# Idempotent: skips the clone if present, and skips `git apply` if the patch is already applied.
set -euo pipefail
: "${VP_NFT_DIR:?source config/env.sh first}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$VP_NFT_DIR/.git" ] || git clone https://github.com/NVlabs/DiffusionNFT.git "$VP_NFT_DIR"
cd "$VP_NFT_DIR"
if git apply --reverse --check "$HERE/diffusionnft_vp.patch" 2>/dev/null; then echo "patch already applied"; exit 0; fi
git checkout -q cbb14f8
git apply "$HERE/diffusionnft_vp.patch"
git --no-pager diff --stat
