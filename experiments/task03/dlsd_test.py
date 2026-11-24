from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import torch, numpy as np, cv2, time
from deeplsd.models.deeplsd_inference import DeepLSD
conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval()
g = cv2.imread(DATA + "/datasets/nyu/rgb512/0000.png", 0)
x = torch.tensor(g[None, None], dtype=torch.float32, device="cuda") / 255
with torch.no_grad():
    out = net({"image": x}); t = time.time()
    for _ in range(10): out = net({"image": x})
torch.cuda.synchronize(); L = out["lines"][0]; print("lines", L.shape, "s/img", (time.time() - t) / 10, L[:2])
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0].reshape(-1, 4); ll = np.hypot(*(lsd[:, 2:] - lsd[:, :2]).T)
dl = L.reshape(-1, 4); dd = np.hypot(*(dl[:, 2:] - dl[:, :2]).T)
print("segments >=20px: LSD", (ll >= 20.48).sum(), "DeepLSD", (dd >= 20.48).sum(), "| total len LSD", ll[ll>=20.48].sum().round(), "DeepLSD", dd[dd>=20.48].sum().round())
