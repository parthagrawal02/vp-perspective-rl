"""New rotation rule: drop at 2 deg significant (one-sided Wilcoxon, p<0.01) AND drop at 4 and at 8 not significantly
smaller than at 2 (paired one-sided Wilcoxon on per-image drop4-drop2 / drop8-drop2, 'less', p<0.01 -> fail)."""
import numpy as np
from scipy.stats import wilcoxon


def _p(x, alt):
    x = x[~np.isnan(x)]
    return float(wilcoxon(x, alternative=alt).pvalue) if len(x) > 10 and np.any(x != 0) else np.nan


def rot_new(df, v, split, winlen):
    d = df[(df.det == "dlsd") & (df.rseed == 0) & (df.split == split)]
    r = d[d.set == "real"].set_index("id")[v]
    keep = [i for i in r.dropna().index if winlen.get(i, 0) >= 100]
    rot = d[d.set == "real_rot"]
    dr = {t: (r.reindex(keep) - rot[rot.theta == t].set_index("id")[v].reindex(keep)).values for t in (2.0, 4.0, 8.0)}
    p2 = _p(dr[2.0], "greater"); p4 = _p(dr[4.0] - dr[2.0], "less"); p8 = _p(dr[8.0] - dr[2.0], "less")
    ok = bool(p2 < 0.01 and not (p4 < 0.01) and not (p8 < 0.01))
    return dict(p_rot2=p2, p_4lt2=p4, p_8lt2=p8, rot_new=ok)
