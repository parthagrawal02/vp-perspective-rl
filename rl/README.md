# rl/: DiffusionNFT + frozen perspective reward (**PROVISIONAL**)

> **Provisional.** The floor reference `Lc_base` used during RL is being fixed: it currently comes from training-style rollouts
> (10 dpm2 steps, deterministic, CFG 1.0) while the reward was validated with CFG 4.5 / 40-step twins (see the top-level README,
> Known issues). Only a 3-epoch smoke test has been run (Task 12). Don't report RL numbers from this setup as final.

We do not copy DiffusionNFT. `diffusionnft_vp.patch` applies to **NVlabs/DiffusionNFT @ `cbb14f8`**:

| file | change |
|---|---|
| `flow_grpo/rewards.py` | new reward `vp_perspective` (calls `vp_reward.final.perspective_reward` with the `lc_base` map `$VP_LC_BASE`, default `dataset/vp_hypersim/lc_base.json`; forkserver pool of `$VP_NWORKERS` geometry workers); returns the reward parts + two high-frequency image statistics (monitoring only, not in R); registered in `multi_score`, parts logged as `vp_<part>` |
| `config/nft.py` | dataset `vp_hypersim` allowed; config `sd3_vp()` = DiffusionNFT's SD3 defaults with `reward_fn={"vp_perspective": 1.0}`, `n_gpus=$NFT_N_GPUS` (default 4) |
| `scripts/train_nft_sd3.py` | `[VP]` logging only: per-epoch reward-part stats (`vp_epoch_stats.csv`, wandb `vp/*`), eval summary (`vp_eval_stats.csv`), timings, peak GPU memory, first 16 rank-0 samples per epoch with their logged reward parts (`samples/epoch_NNNN/`) |
| `scripts/evaluation.py` | removed the two `unifiedreward` entries (needs an sglang server we don't run) |

The integration differs from what ran in Task 12 in one way: `rewards.py` no longer adds `../vp_reward` and `../DeepLSD` to
`sys.path`, because both are pip-installed now. DiffusionNFT is Apache-2.0; `LICENSE-DiffusionNFT-Apache-2.0.txt` is its license
(the top-level README explains what it allows).

**Reward version.** `VP_REWARD=v1` (default) trains on the frozen reward; `VP_REWARD=v2` on the experimental
`vp_reward/reward_v2.py` (frozen geometry + an upper structure cap against densification; not frozen, see
`experiments/task15/NOTES.md`). `rl/lcbase.py --sampler validated` builds the floor reference with the validation-twin sampler
(40 steps, CFG 4.5) into `lc_base_validated.json`. **NCSA Delta:** `rl/delta/README.md` (setup, Slurm jobs, budget).

## Steps
```bash
source config/env.sh
bash rl/apply_patch.sh                                   # (setup_nft_vp.sh already did this)
python data/hypersim_train_dl.py; python data/prep_hypersim_train.py; GPU=1 bash data/run_caption.sh hypersim_train   # training prompts
python rl/make_dataset.py                                # -> $VP_NFT_DIR/dataset/vp_hypersim/{train,test}.txt, prompt_map.json
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=<free gpu k> python rl/lcbase.py $k 4 & done; wait; python rl/lcbase.py merge   # lc_base.json
GPUS=1,3,5,7 EPOCHS=3 RUN=smoke bash rl/run_train.sh     # check nvidia-smi first
```
Task 12 numbers: train 1,921 prompts / test 596 (fresh split); `lc_base.json` 2,517 prompts × 4 samples (~20 min on 4 GPUs);
smoke test 4 GPUs, micro-batch 3 × 96 accumulation (default 6 × 48 OOMs on 48 GB), ~441 s/epoch (sampling 107, reward wait 1,
training 333), eval 199 s, peak 44.6 GB/GPU allocated. The wiring check (`experiments/task12/check12.py`) found the logged reward
parts identical to an offline re-score of the same pixels (R max diff 0.0014 from PickScore batch numerics).
