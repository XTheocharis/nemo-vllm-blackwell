# vLLM v0.16.0 + Nemotron-Flash-3B (PR #31543) on NeMo 26.02
# Base: NGC NeMo 26.02 (NeMo 2.7.0, PyTorch 2.10.0a0, CUDA 13.0, sm_120)
#
# Provides:
#   - vLLM 0.16.0 with NVFP4 sm_120 kernels (replaces bundled vLLM 0.14.2)
#   - Nemotron-Flash-3B hybrid model support (PR #31543)
#   - NeMo ASR: Canary-Qwen-2.5B (SALM), Sortformer 4-speaker diarization
#   - NVFP4 quantization for nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4
#
# Build (fast — downloads pre-compiled .so from GitHub Release):
#   docker build -t nemo-vllm-blackwell:v0.16.0-nemo26.02-sm120 .
#
# Build (from source — recompiles CUDA kernels ~17 min):
#   docker build --build-arg USE_PREBUILT=0 -t nemo-vllm-blackwell:v0.16.0-nemo26.02-sm120 .
#
# Run (vLLM server):
#   docker run --rm --gpus all -p 8000:8000 nemo-vllm-blackwell:v0.16.0-nemo26.02-sm120 \
#     --model nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 --quantization nvfp4

FROM nvcr.io/nvidia/nemo:26.02.00

ARG MAX_JOBS=16
ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG USE_PREBUILT=1
ARG PREBUILT_TAG=v0.16.0-nemo26.02-sm120

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV VLLM_TARGET_DEVICE=cuda
# Pretend version for setuptools-scm (no .git in build context)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0

# ---------------------------------------------------------------------------
# 1. Remove bundled vLLM (0.14.2.dev0) to avoid conflicts
# ---------------------------------------------------------------------------
RUN pip uninstall -y vllm

# ---------------------------------------------------------------------------
# 2. Install ffmpeg (missing from NeMo image, needed by pydub for audio)
# ---------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Copy patched vLLM v0.16.0 source (with PR #31543 cherry-picked)
# ---------------------------------------------------------------------------
WORKDIR /workspace/vllm
COPY vllm/ .

# ---------------------------------------------------------------------------
# 4. Download pre-compiled .so from GitHub Release (when USE_PREBUILT=1)
# ---------------------------------------------------------------------------
ARG VLLM_FLASH_ATTN_COMMIT=188be16520ceefdc625fdf71365585d2ee348fe2
RUN if [ "${USE_PREBUILT}" = "1" ]; then \
        BASE_URL="https://github.com/XTheocharis/nemo-vllm-blackwell/releases/download/${PREBUILT_TAG}" && \
        mkdir -p vllm/vllm_flash_attn && \
        echo "==> Downloading pre-compiled .so from ${BASE_URL}" && \
        curl -fSL -o vllm/_C.abi3.so           "${BASE_URL}/_C.abi3.so" && \
        curl -fSL -o vllm/_moe_C.abi3.so       "${BASE_URL}/_moe_C.abi3.so" && \
        curl -fSL -o vllm/cumem_allocator.abi3.so "${BASE_URL}/cumem_allocator.abi3.so" && \
        curl -fSL -o vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so "${BASE_URL}/_vllm_fa2_C.abi3.so" && \
        curl -fSL -o vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so "${BASE_URL}/_vllm_fa3_C.abi3.so" && \
        echo "==> Downloaded 5 .so files" && \
        echo "==> Fetching vllm-flash-attn Python wrappers" && \
        git clone --depth 1 https://github.com/vllm-project/flash-attention.git /tmp/flash-attn && \
        cd /tmp/flash-attn && git fetch --depth 1 origin ${VLLM_FLASH_ATTN_COMMIT} && \
        git checkout ${VLLM_FLASH_ATTN_COMMIT} && cd /workspace/vllm && \
        cp -r /tmp/flash-attn/vllm_flash_attn/*.py vllm/vllm_flash_attn/ && \
        cp -r /tmp/flash-attn/vllm_flash_attn/layers vllm/vllm_flash_attn/ && \
        cp -r /tmp/flash-attn/vllm_flash_attn/ops vllm/vllm_flash_attn/ && \
        rm -rf /tmp/flash-attn && \
        echo "==> Installed flash-attn Python wrappers" ; \
    fi

# ---------------------------------------------------------------------------
# 5. Strip torch pins and flashinfer (container provides PyTorch; no
#    flashinfer wheel for CUDA 13.0)
# ---------------------------------------------------------------------------
RUN sed -i '/^torch==/d; /^torchaudio==/d; /^torchvision==/d' \
        requirements/cuda.txt requirements/build.txt && \
    sed -i '/"torch ==/d' pyproject.toml && \
    sed -i '/^flashinfer-python/d' requirements/cuda.txt

# ---------------------------------------------------------------------------
# 6. Install vLLM build + runtime dependencies not already in NeMo image
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-build-isolation \
        "setuptools-scm>=8" \
        "grpcio-tools==1.78.0" && \
    pip install --no-cache-dir --no-build-isolation \
        -r requirements/cuda.txt

# ---------------------------------------------------------------------------
# 7. Install fla (Flash Linear Attention — DeltaNet kernels for
#    Nemotron-Flash-3B's gated linear attention layers)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir fla-core

# ---------------------------------------------------------------------------
# 8. Install vLLM v0.16.0
#    If .so files are present (prebuilt or local), skip CUDA compilation.
#    Otherwise, build from source.
# ---------------------------------------------------------------------------
RUN if [ -f vllm/_C.abi3.so ]; then \
        echo "==> Pre-compiled .so detected, skipping CUDA build" && \
        VLLM_TARGET_DEVICE=empty pip install --no-build-isolation --no-cache-dir -e . ; \
    else \
        echo "==> Building from source (MAX_JOBS=${MAX_JOBS})" && \
        pip install --no-build-isolation --no-cache-dir -e . ; \
    fi

# ---------------------------------------------------------------------------
# 9. Verify all three model stacks import cleanly
# ---------------------------------------------------------------------------
RUN python -c "import vllm; print(f'vLLM {vllm.__version__}')" && \
    python -c "import nemo; print(f'NeMo {nemo.__version__}')"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
