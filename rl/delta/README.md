# Running the RL fine-tune on NCSA Delta

Delta's **A40 (48 GB)** nodes match the Task 12 smoke test (4 × 48 GB, micro-batch 3 × 96) and are the cheapest GPUs
(batch 0.5× per GPU-hour). Everything below runs on Delta (`login.delta.ncsa.illinois.edu`), not DeltaAI: DeltaAI is aarch64,
and the pytlsd build plus the pinned cu128 wheels are only tested on x86.

| file | what |
|---|---|
| `env.delta.sh` | Delta paths: `VP_DATA=/work/hdd/bjop/$USER/vp-data` (shared course quota, not backed up), env in `~/envs/nft-vp` |
| `setup.sh` | login node, once: `module load pytorch-conda`, `envs/setup_nft_vp.sh` with conda, pre-fetch HF models to `/work` |
| `check.slurm` | 1 A40, ≤ 30 min: CUDA, unit tests, both reward versions end to end |
| `lcbase.slurm` | 4 A40: floor reference `Lc_base` per prompt (`SAMPLER=validated` default, or `train`) |
| `train.slurm` | 4 A40: `rl/run_train.sh` with `VP_REWARD=v1|v2`, `EPOCHS`, `RUN`, `VP_LC_BASE` |

## Steps

```bash
# laptop: code only (the repo has no data); vp_pairs_v1 is needed only for the fallback prompts below
rsync -av --exclude .venv --exclude third_party ./ NETID@login.delta.ncsa.illinois.edu:~/vp-perspective-rl/
rsync -av ~/Downloads/vp_pairs_v1/{metadata.csv,captions.jsonl} NETID@login.delta.ncsa.illinois.edu:/work/hdd/bjop/NETID/vp_pairs_v1/

# Delta login node
cd ~/vp-perspective-rl
bash rl/delta/setup.sh           # env + DeepLSD + patched DiffusionNFT + weights + HF prefetch
# first run stops at the gated SD3.5 download: accept its license on huggingface.co, then
source config/env.sh && $VP_ENV/bin/huggingface-cli login && bash rl/delta/setup.sh
sbatch rl/delta/check.slurm      # expect CHECK_OK in slurm-vp-check-<id>.out

# prompts: preferred = the Task 12 set (copy dataset/vp_hypersim/{train,test}.txt from the project machine into
#          $VP_NFT_DIR/dataset/vp_hypersim/); fallback = from vp_pairs_v1 (see that script's docstring for the differences)
source config/env.sh && $VP_ENV/bin/python rl/make_dataset_from_pairs.py /work/hdd/bjop/$USER/vp_pairs_v1
#   -> train.txt (SELECT), test.txt (fresh = validation during training), final_test.txt (REPORT, evaluate once);
#      keeps only pairs whose reference image beats its base twins (--filter none to keep all)

sbatch rl/delta/lcbase.slurm                                         # Lc_base, validated sampler (resumable)
EPOCHS=3 RUN=smoke VP_REWARD=v1 sbatch --time=01:00:00 rl/delta/train.slurm   # smoke test
EPOCHS=150 RUN=v2_run1 VP_REWARD=v2 sbatch rl/delta/train.slurm               # real run (24 h limit; ≤ 48 h allowed)
squeue -u $USER;  tail -f slurm-vp-nft-<id>.out
```
Outputs: `$VP_DATA/outputs/rl/<RUN>/nft_logs/` (`vp_epoch_stats.csv`, `vp_eval_stats.csv`, `samples/epoch_NNNN/`),
checkpoints `$VP_DATA/checkpoints/rl/<RUN>/`. `WANDB_MODE` defaults to offline.

## Don't auto-load the pytorch-conda module
The NCSA guide suggests adding `module load pytorch-conda; conda activate ...` to `~/.bashrc`. **Don't, for this project**:
that module puts NCSA's PyTorch 2.8 env on the library path, which breaks the pytlsd build and can shadow the env's torch 2.7.1.
Our scripts call `$VP_ENV/bin/python` directly and never need an activated env.

## Budget (shared by the whole class)
A40 batch = 0.5 units per GPU-hour, so one 4-GPU node costs 2 units/hour: smoke test ~1-2, `lcbase` (validated, ~10k images)
roughly 6-12, a 24 h training run ~48. Run `accounts` before submitting. Debug with the smoke test, not the 24 h job.

## What to watch during training
`vp_epoch_stats.csv` per epoch: `R_mean` should rise. Treat these as hacking alarms: `Lc_mean / Lc_base_mean` climbing
(densification; with v2, `cap_activity`), `hf_noise_mean` / `hf_lapvar_mean` rising (the σ=2 noise loophole), `pickscore` /
`aesthetic` falling. Look at `samples/epoch_NNNN/` every few epochs: the Task 12 smoke test already looked grid-heavy by epoch 2.
