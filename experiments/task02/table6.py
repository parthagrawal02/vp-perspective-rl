from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, numpy as np, pandas as pd
OUT = DATA + "/outputs/task2"
sel = json.load(open(f"{OUT}/selection.json")); pick = sel["pick"] or {"cfg": "L4_t1.5", "cand": "RegMean"}; L = sel["lref"][pick["cfg"]]; K = pick["cand"]
df = pd.read_pickle(f"{OUT}/scores.pkl"); m = df[(df.rseed == 0) & (df.cfg == pick["cfg"]) & (df.split == "REPORT")].copy()
m["seed"] = m["seed"].astype(int); m["gated"] = m[K] * np.minimum(1, m.Lc / L)
pa = pd.read_pickle(f"{OUT}/pick_aes.pkl"); pa["seed"] = pa["seed"].astype(int)
m = m.merge(pa, on=["set", "id", "seed"], how="left")
rows = {"Real NYU photos": m[m.set == "real"],
        "SD3.5-M base (seeds 0-1)": m[(m.set == "base") & (m.seed <= 1)],
        "Base + perspective suffix": m[m.set == "base_suffix"],
        f"Best-of-4 by {K} (gated) †": m[m.set == "base"].sort_values("gated").groupby("id").tail(1),
        "Flow-GRPO PickScore": m[m.set == "flowgrpo"],
        "DiffusionNFT multi": m[m.set == "nft"]}
real = m[m.set == "real"].set_index("id")["gated"]
out = []
for name, d in rows.items():
    r = {"row": name, "n": len(d), f"{K}·gate": d.gated.mean(), K: d[K].mean()}
    for c in ["G", "NM", "RegMean", "RegMax"]:
        if c != K: r[c] = d[c].mean()
    r.update(Lc=d.Lc.mean(), PickScore=d.pickscore.mean(), Aesthetic=d.aesthetic.mean())
    if name.startswith("Real"): r["real_winrate"] = np.nan
    else:
        w = [np.mean((real[i] > g.gated.values) + 0.5 * (real[i] == g.gated.values)) for i, g in d.groupby("id")]
        r["real_winrate"] = np.mean(w)
    out.append(r)
T = pd.DataFrame(out); T.to_csv(f"{OUT}/table6.csv", index=False)
pd.set_option("display.width", 250); print(f"score={K} cfg={pick['cfg']} L_ref={L:.0f}  ({'PROVISIONAL' if sel['pick'] is None else 'chosen'})")
print(T.round(3).to_string(index=False))
