# Distortion + border checks for a reference image set. Usage: refcheck.py <glob> <n>
import sys, glob, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
P = sorted(glob.glob(sys.argv[1])); P = P[:: max(1, len(P) // int(sys.argv[2]))]
lsd = None
def longlen(g):
    global lsd
    lsd = lsd or cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    l = lsd.detect(g)[0]
    if l is None: return 0.0
    l = l.reshape(-1, 4); L = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1]); return float(L[L >= 0.15 * min(g.shape)].sum())
KS = [-0.12, -0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 0.12]
def job(p):
    cv2.setNumThreads(1)
    im = cv2.GaussianBlur(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (0, 0), 1.0); h, w = im.shape; f = 0.8 * max(h, w)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]); out = []
    m = int(0.06 * min(h, w))
    for k in KS:
        und = cv2.remap(im, *cv2.initUndistortRectifyMap(K, np.array([k + 1e-9, 0, 0, 0, 0]), None, K, (w, h), cv2.CV_32FC1), cv2.INTER_LINEAR)
        out.append(longlen(und[m:-m, m:-m]))
    rows, cols = im.astype(float).mean(1), im.astype(float).mean(0)
    return out, rows[:14].tolist() + rows[-14:].tolist(), cols[:14].tolist() + cols[-14:].tolist(), im.shape
with ProcessPoolExecutor(64) as ex: R = list(ex.map(job, P))
A = np.array([r[0] for r in R]); A = A / np.maximum(A[:, KS.index(0.0)][:, None], 1)
print(f"n={len(P)} size={R[0][3]}  long-line length vs radial k (relative to k=0, focal=0.8*max side):")
for k, v, w in zip(KS, A.mean(0), (A.argmax(1)[:, None] == np.arange(len(KS))).mean(0)): print(f"  k={k:+.2f}: mean rel {v:.3f}  argmax share {w:.2f}")
print("mean row intensity first/last 8:", np.round(np.mean([r[1] for r in R], 0), 1))
print("mean col intensity first/last 8:", np.round(np.mean([r[2] for r in R], 0), 1))
print("frac imgs with pure-black (<3) first row or col:", np.mean([(min(r[1][0], r[1][-1], r[2][0], r[2][-1]) < 3) for r in R]))
