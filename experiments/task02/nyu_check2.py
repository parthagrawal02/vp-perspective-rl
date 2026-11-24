from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import h5py, numpy as np, cv2
f = h5py.File(DATA + "/datasets/nyu/raw/nyu_depth_v2_labeled.mat", "r"); I = f["images"]
K = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02], [0, 5.1946961112127485e+02, 2.5373616633400465e+02], [0, 0, 1]])
D0 = np.array([2.0796615318809061e-01, -5.8613825163911781e-01, 7.2231363135888329e-04, 1.0479627195765181e-03, 4.9856986684705107e-01])
idx = np.arange(3, 1449, 7); imgs = [I[i].transpose(2, 1, 0) for i in idx]
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
ys, xs = np.mgrid[0:480, 0:640].astype(np.float32)
def m_undist(a):  # a*dist: output undistorted (for a>0), a<0 -> apply distortion
    if a >= 0: return cv2.initUndistortRectifyMap(K, D0 * a, None, K, (640, 480), cv2.CV_32FC1)
    u = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1)[:, None], K, D0 * (-a), P=K).reshape(480, 640, 2)
    return u[..., 0].copy(), u[..., 1].copy()
maps = {a: m_undist(a) for a in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]}
maps["shift0.5"] = (xs + 0.5, ys + 0.5)
# displacement magnitude of the undistortion (inside the crop window)
mx, my = maps[1.0]; dmag = np.hypot(mx - xs, my - ys)[44:471, 40:601]
print(f"undistortion displacement inside crop: median {np.median(dmag):.2f}px, max {dmag.max():.2f}px")
def stats(im):
    g = cv2.cvtColor(im[50:465, 46:595], cv2.COLOR_RGB2GRAY)
    l = lsd.detect(g)[0].reshape(-1, 4); L = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    return L[L >= 100].sum(), L[L >= 150].sum(), L[L >= 20].sum()
out = {}
for k, mp in maps.items():
    out[k] = np.array([stats(cv2.remap(im, mp[0], mp[1], cv2.INTER_LINEAR)) for im in imgs])
out["orig(no remap)"] = np.array([stats(im) for im in imgs])
for k, v in out.items():
    print(f"{str(k):16s} len>=100: {v[:,0].mean():7.1f}  len>=150: {v[:,1].mean():7.1f}  len>=20: {v[:,2].mean():7.1f}")
