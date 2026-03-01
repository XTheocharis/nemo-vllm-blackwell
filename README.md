# nemo-vllm-blackwell

Docker build for **vLLM 0.16.0** on **NeMo 26.02** targeting **NVIDIA Blackwell GPUs** (sm_120).

Combines three model stacks in a single image:

| Model | Framework | Purpose |
|-------|-----------|---------|
| [Nemotron-Nano-9B-v2-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4) | vLLM (NVFP4) | LLM inference |
| [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) | NeMo SALM | Speech-to-text |
| [Sortformer 4-speaker](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | NeMo ASR | Speaker diarization |
| [Nemotron-Flash-3B](https://huggingface.co/nvidia/Nemotron-Flash-3B) | vLLM (PR [#31543](https://github.com/vllm-project/vllm/pull/31543)) | Hybrid Mamba2+DeltaNet+Transformer |

## What's in the image

- **Base**: `nvcr.io/nvidia/nemo:26.02.00` (NeMo 2.7.0, PyTorch 2.10.0a0, CUDA 13.0)
- **vLLM 0.16.0** with NVFP4 sm_120 CUDA kernels (replaces bundled vLLM 0.14.2)
- **PR #31543** cherry-picked for Nemotron-Flash-3B hybrid model support
- **fla-core** for DeltaNet gated linear attention kernels
- **ffmpeg** for audio processing

## Quick start

```bash
git clone --recursive https://github.com/XTheocharis/nemo-vllm-blackwell.git
cd nemo-vllm-blackwell
docker build -t vllm-nvfp4 .
```

By default, the build downloads pre-compiled CUDA kernels from [GitHub Releases](https://github.com/XTheocharis/nemo-vllm-blackwell/releases) and skips compilation. Total build time is ~2 minutes (mostly pip installs).

### Run the vLLM server

```bash
docker run --rm --gpus all -p 8000:8000 vllm-nvfp4 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 \
    --quantization nvfp4
```

## Build options

| Build arg | Default | Description |
|-----------|---------|-------------|
| `USE_PREBUILT` | `1` | `1` = download pre-compiled `.so` from GitHub Release. `0` = compile from source. |
| `PREBUILT_TAG` | `v0.16.0-nemo26.02-sm120` | GitHub Release tag to download pre-compiled artifacts from. |
| `MAX_JOBS` | `16` | Parallel compilation jobs (only used when building from source). |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | CUDA architectures to compile for (only used when building from source). |

### Build from source

To recompile all CUDA kernels from source (~17 minutes on 16 cores):

```bash
docker build --build-arg USE_PREBUILT=0 -t vllm-nvfp4 .
```

### Extract new pre-compiled artifacts

After a from-source build, extract the `.so` files to avoid recompiling next time:

```bash
docker create --name vllm-extract vllm-nvfp4
for f in vllm/_C.abi3.so vllm/_moe_C.abi3.so vllm/cumem_allocator.abi3.so \
         vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so; do
    docker cp "vllm-extract:/workspace/vllm/$f" "vllm/$f"
done
docker rm vllm-extract
```

These can then be uploaded as a new GitHub Release for future builds.

## Repository structure

```
nemo-vllm-blackwell/
├── Dockerfile              # Multi-path build (prebuilt or from-source)
├── download-prebuilt.sh    # Optional: download .so locally without Docker
├── vllm/                   # Git submodule → XTheocharis/vllm @ v0.16.0-nemotron-flash
│   ├── vllm/               # vLLM Python package
│   ├── csrc/               # CUDA kernel sources
│   ├── requirements/       # Python dependencies
│   └── ...
└── README.md
```

The `vllm/` submodule points to a [fork of vLLM](https://github.com/XTheocharis/vllm/tree/v0.16.0-nemotron-flash) on the `v0.16.0-nemotron-flash` branch — this is vLLM v0.16.0 with 11 commits from [PR #31543](https://github.com/vllm-project/vllm/pull/31543) cherry-picked for Nemotron-Flash-3B support.

## Pre-compiled artifacts

The [GitHub Release](https://github.com/XTheocharis/nemo-vllm-blackwell/releases/tag/v0.16.0-nemo26.02-sm120) contains 5 pre-compiled `.so` files (1.7 GB total):

| File | Size | Contents |
|------|------|----------|
| `_C.abi3.so` | 310 MB | Core custom ops + NVFP4 kernels |
| `_moe_C.abi3.so` | 81 MB | Mixture-of-experts kernels |
| `cumem_allocator.abi3.so` | 105 KB | CUDA memory allocator |
| `_vllm_fa2_C.abi3.so` | 281 MB | Flash Attention 2 |
| `_vllm_fa3_C.abi3.so` | 1003 MB | Flash Attention 3 |

These were compiled against NeMo 26.02 (CUDA 13.0, PyTorch 2.10.0a0) with `TORCH_CUDA_ARCH_LIST="12.0"` (Blackwell sm_120). They are **not** portable to other CUDA versions or GPU architectures — rebuild from source if changing the base image.

## Target hardware

Tested on **NVIDIA RTX 5070 Ti** (16 GB VRAM, Blackwell sm_120).

Requires:
- NVIDIA driver R570+ (for CUDA 12.8) or R580+ (for CUDA 13.0)
- Docker with `--gpus` support (nvidia-container-toolkit)

## License

vLLM is licensed under [Apache 2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE). NeMo components are subject to [NVIDIA's license terms](https://developer.nvidia.com/nemo-microservices).
