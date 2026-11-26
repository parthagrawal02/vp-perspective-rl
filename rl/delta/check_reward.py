"""End-to-end check of both reward versions on one GPU (used by rl/delta/check.slurm). Uses the forkserver start method, as
flow_grpo/rewards.py does: geometry worker pools forked after CUDA init deadlock (Task 12 notes)."""
import multiprocessing
import time

import numpy as np


def main():
    from vp_reward import final, reward_v2
    rng = np.random.default_rng(0)
    ims = rng.integers(0, 255, (4, 512, 512, 3), dtype=np.uint8)
    ims[:, 100:400, 250:254] = 20; ims[:, 250:254, 50:450] = 20      # a little structure
    prompts = ["A photo of a kitchen"] * 4
    for name, mod in (("v1", final), ("v2", reward_v2)):
        t0 = time.time()
        fn = mod.perspective_reward(device="cuda", lc_base={prompts[0]: 5000.0}, n_workers=4)
        t1 = time.time()
        R, info = fn(ims, prompts)
        print(f"{name}: load {t1 - t0:.0f}s, score {time.time() - t1:.1f}s, R {np.round(R, 3)}, keys {sorted(info)}", flush=True)
    print("CHECK_OK", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
