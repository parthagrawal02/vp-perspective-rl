# Environments

Two envs, both created with micromamba `-p` under `$VP_DATA/envs/` (never in `~`):

| env | for | python / torch | spec | setup |
|---|---|---|---|---|
| `nft-vp` | the reward, eval, RL training, twin generation, all experiments | 3.10.16 / **2.7.1+cu128** | `requirements-nft-vp.lock.txt` | `bash envs/setup_nft_vp.sh` |
| `vlm` | captioning only (`data/caption.py`, Qwen3.8-27B-FP8 on vLLM) | 3.12.16 / 2.14.0+cu130, vLLM 0.33.0 | `requirements-vlm.lock.txt` | `bash envs/setup_vlm.sh` |

Always `source config/env.sh` first (copy it from `config/env.example.sh`): it puts every cache (`HF_HOME`, pip, torch, triton,
conda pkgs, vLLM / FlashInfer JIT) and `TMPDIR` under `$VP_DATA`, and puts the DiffusionNFT clone on `PYTHONPATH`.

## nft-vp, step by step (what `setup_nft_vp.sh` does)
1. `micromamba create -p $VP_ENV -c conda-forge python=3.10.16`
2. `pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128`
3. `pip install --no-deps -r envs/requirements-nft-vp.lock.txt`. This is the exact `pip freeze` of the env that produced every
   result (diffusers 0.33.1, transformers 4.40.0, peft 0.10.0, accelerate 1.4.0, numpy 1.26.4, opencv 4.11.0.86, ...), minus torch
   and the locally built packages below. `--no-deps` is deliberate: DiffusionNFT's setup.py pins torch 2.6.0, and pip would
   otherwise try to downgrade torch.
4. DeepLSD @ `f7d9d62` (clone) + **pytlsd built from DeepLSD's submodule** (commit `21381ca`, reports version 0.0.3), then `pip install -e DeepLSD --no-deps`.
5. `rl/apply_patch.sh`: clone NVlabs/DiffusionNFT, check out `cbb14f8`, apply `rl/diffusionnft_vp.patch`; `pip install -e --no-deps`.
6. `pip install -e vp_reward --no-deps`.
7. Weights: DeepLSD `deeplsd_wireframe.tar` (MIT) into `$VP_DATA/models/deeplsd/`; aesthetic head `sac+logos+ava1-l14-linearMSE.pth`
   into `$VP_DATA/models/reward_ckpts/` (symlinked as `$VP_NFT_DIR/reward_ckpts`). PickScore and CLIP come from the HF hub on first use.
   SD3.5-Medium (only needed for generation / training) is gated: accept its license on huggingface.co and `huggingface-cli login`.

Expected `pip check` notes (harmless): diffusion-nft wants torch 2.6.0 / torchvision 0.21.0, flash-attn, xformers, deepspeed (none
are used by our code paths).

## Notes
- **Blackwell (sm_120, RTX PRO 5000 / RTX 50xx) needs a CUDA 12.8+ PyTorch build.** torch 2.6.0+cu126 (upstream's pin) has no sm_120
  kernels. The driver must support CUDA 12.8 (we used 580.105.08 / CUDA 13.0). No system CUDA toolkit is needed for nft-vp.
- **pytlsd build workaround.** pytlsd isn't on PyPI at the version DeepLSD uses, so it is built from source. Its build pulls the newest
  `cmake` (4.x), which refuses the `cmake_minimum_required` of the bundled pybind11 (v2.6.2). Fix: `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
  We build from a copy outside the clone so no `build/` directory lands in it.
- `flow_grpo/` (DiffusionNFT) has no `__init__.py`, so `pip install -e` exposes nothing: `config/env.sh` adds `$VP_NFT_DIR` to
  `PYTHONPATH` (and `vp_reward.score` adds it itself when it needs the quality scorers).
- Process pools inside the reward must use the **forkserver** start method: a fork-started pool deadlocks after CUDA / OpenMP init.
  `config/env.sh` also sets `OMP/MKL/OPENBLAS_NUM_THREADS=1` (avoids oversubscription with many workers).
- vLLM on sm_120 (`data/run_caption.sh` sets these): `CUDA_HOME=$VP_VLM_ENV/lib/python3.12/site-packages/nvidia/cu13` (FlashInfer JIT
  needs nvcc from the wheel), env `bin/` on `PATH` (ninja), `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's sampling kernel fails
  to JIT on sm_120), `max_num_seqs=128`.
- History: Tasks 3–11 used a separate `deeplsd` env (python 3.10, torch 2.7.1+cu128, pytlsd, DeepLSD) for line detection. Since
  Task 12, `nft-vp` holds both, and the reproduction check gives identical G and Lc in it.
