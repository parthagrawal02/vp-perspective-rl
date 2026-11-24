from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import h5py, numpy as np, cv2, sys

f = h5py.File(DATA + "/datasets/nyu/raw/nyu_depth_v2_labeled.mat", "r")
I = f["images"]
K = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02], [0, 5.1946961112127485e+02, 2.5373616633400465e+02], [0, 0, 1]])
dist = np.array([2.0796615318809061e-01, -5.8613825163911781e-01, 7.2231363135888329e-04, 1.0479627195765181e-03, 4.9856986684705107e-01])
idx = np.arange(0, 1449, 7)  # 207 images
imgs = np.stack([I[i].transpose(2, 1, 0) for i in idx])  # (n,480,640,3)
# 1) border analysis: per-row / per-col std of mean over images
g = imgs.mean(3)
print("mean intensity of rows 0..10 / 470..479 (avg over imgs):", np.round(g.mean((0, 2))[:10], 1), np.round(g.mean((0, 2))[-10:], 1))
print("mean intensity of cols 0..10 / 630..639:", np.round(g.mean((0, 1))[:10], 1), np.round(g.mean((0, 1))[-10:], 1))
# find first/last row/col where pixel std across image (within row) > threshold, per image
def border(gi, thr=3.0):
    rs = gi.std(1); cs = gi.std(0)
    r = np.where(rs > thr)[0]; c = np.where(cs > thr)[0]
    return r[0], r[-1], c[0], c[-1]
B = np.array([border(gi) for gi in g])
print("content bbox (row0,row1,col0,col1) min/median/max over imgs:\n", B.min(0), np.median(B, 0), B.max(0))
# pixels outside toolbox mask that are exactly constant
m = np.zeros((480, 640), bool); m[44:471, 40:601] = True
print("frac of near-white/near-black/const pixels outside mask:", np.mean((g[:, ~m] > 250) | (g[:, ~m] < 5)))
# 2) straightness test: long-segment length for original / undistorted-again / distorted
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
newK = K.copy()
map_u = cv2.initUndistortRectifyMap(K, dist, None, newK, (640, 480), cv2.CV_32FC1)
# "distort" = inverse: for each output pixel (distorted coords) find source in undistorted image
ys, xs = np.mgrid[0:480, 0:640].astype(np.float32)
pts = np.stack([xs.ravel(), ys.ravel()], 1)[:, None]
und = cv2.undistortPoints(pts, K, dist, P=newK).reshape(480, 640, 2)
map_d = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))
def longlen(im, minlen=100):
    gg = cv2.cvtColor(im[44:471, 40:601], cv2.COLOR_RGB2GRAY)
    l = lsd.detect(gg)[0]
    if l is None: return 0, 0
    l = l.reshape(-1, 4); L = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    return L[L >= minlen].sum(), (L >= minlen).sum()
res = {"orig": [], "undistort_again": [], "distort": []}
for im in imgs:
    res["orig"].append(longlen(im))
    res["undistort_again"].append(longlen(cv2.remap(im, *map_u, cv2.INTER_LINEAR)))
    res["distort"].append(longlen(cv2.remap(im, *map_d, cv2.INTER_LINEAR)))
for k, v in res.items():
    v = np.array(v); print(f"{k:16s} mean total length of segments>=100px: {v[:,0].mean():.1f}  mean count: {v[:,1].mean():.2f}")
o = np.array(res["orig"])[:, 0]
for k in ("undistort_again", "distort"):
    v = np.array(res[k])[:, 0]; print(f"orig > {k} in {np.mean(o > v)*100:.1f}% of images (ties {np.mean(o==v)*100:.1f}%)")
