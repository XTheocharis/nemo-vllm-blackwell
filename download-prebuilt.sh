#!/usr/bin/env bash
# Download pre-compiled vLLM CUDA kernels from GitHub Release artifacts.
# These .so files were built against NeMo 26.02 (CUDA 13.0, PyTorch 2.10.0a0)
# with TORCH_CUDA_ARCH_LIST="12.0" (Blackwell sm_120).
#
# Usage:
#   ./download-prebuilt.sh          # download latest release
#   ./download-prebuilt.sh v0.16.0-nemo26.02-sm120   # specific tag

set -euo pipefail

REPO="XTheocharis/nemo-vllm-blackwell"
TAG="${1:-v0.16.0-nemo26.02-sm120}"
VLLM_DIR="$(cd "$(dirname "$0")/vllm" && pwd)"

SO_FILES=(
    "_C.abi3.so"
    "_moe_C.abi3.so"
    "cumem_allocator.abi3.so"
    "_vllm_fa2_C.abi3.so"
    "_vllm_fa3_C.abi3.so"
)

# Map filename → destination relative to vllm submodule root
declare -A DEST_MAP=(
    ["_C.abi3.so"]="vllm/_C.abi3.so"
    ["_moe_C.abi3.so"]="vllm/_moe_C.abi3.so"
    ["cumem_allocator.abi3.so"]="vllm/cumem_allocator.abi3.so"
    ["_vllm_fa2_C.abi3.so"]="vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
    ["_vllm_fa3_C.abi3.so"]="vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"
)

echo "Downloading pre-compiled .so files from ${REPO} release ${TAG}..."
echo "Destination: ${VLLM_DIR}"
echo

# Check gh CLI is available
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI not found. Install from https://cli.github.com/" >&2
    exit 1
fi

# Ensure vllm_flash_attn directory exists
mkdir -p "${VLLM_DIR}/vllm/vllm_flash_attn"

# Download each artifact
for file in "${SO_FILES[@]}"; do
    dest="${VLLM_DIR}/${DEST_MAP[$file]}"
    echo "  ${file} -> ${dest}"
    gh release download "${TAG}" \
        --repo "${REPO}" \
        --pattern "${file}" \
        --output "${dest}" \
        --clobber
done

echo
echo "Done. Total size:"
du -sh "${VLLM_DIR}"/vllm/*.so "${VLLM_DIR}"/vllm/vllm_flash_attn/*.so 2>/dev/null
echo
echo "Ready to build: docker build -t vllm-nvfp4 ."
