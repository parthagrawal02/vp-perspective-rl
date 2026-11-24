"""Image perturbations for validation: sharpness-matching degradation (5b) and feathered regional rotation (5c)."""
import cv2
import numpy as np

S = 512


def laplacian_var(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def degrade(img, down=2.0, blur_sigma=0.5, jpeg_q=75):
    """Downscale by `down` (area) + upscale (bilinear), Gaussian blur, JPEG round-trip."""
    h, w = img.shape[:2]
    if down > 1:
        small = cv2.resize(img, (max(1, round(w / down)), max(1, round(h / down))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


WIN = 256  # window side: 1/4 of the image area
FEATHER = 32  # px raised-cosine falloff inside the window border


def choose_window(seg_px, lmin=0.04 * S, stride=64):
    """Top-left (x0,y0) of the WIN x WIN window whose inner region holds the most segment length."""
    seg = np.asarray(seg_px).reshape(-1, 4)
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg, L = seg[L >= lmin], L[L >= lmin]
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    best, arg = -1, (128, 128)
    for y0 in range(0, S - WIN + 1, stride):
        for x0 in range(0, S - WIN + 1, stride):
            inn = (mid[:, 0] > x0 + FEATHER) & (mid[:, 0] < x0 + WIN - FEATHER) & (mid[:, 1] > y0 + FEATHER) & (mid[:, 1] < y0 + WIN - FEATHER)
            v = L[inn].sum()
            if v > best:
                best, arg = v, (x0, y0)
    return arg, float(best)


def window_mask(x0, y0):
    m = np.zeros((S, S), np.float32)
    r = np.arange(WIN, dtype=np.float32)
    edge = np.minimum(r, WIN - 1 - r)
    ramp = np.clip(edge / FEATHER, 0, 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)  # raised cosine 0 -> 1
    m[y0 : y0 + WIN, x0 : x0 + WIN] = ramp[:, None] * ramp[None, :]
    return m


def rotate_region(img, x0, y0, theta_deg, back_and_forth=False):
    """Rotate the content of the window about its center by theta, feathered blend into the original.
    back_and_forth=True rotates by +theta then -theta (same resampling/blending, no net geometry change): control."""
    c = (x0 + WIN / 2 - 0.5, y0 + WIN / 2 - 0.5)
    M = cv2.getRotationMatrix2D(c, theta_deg, 1.0)
    rot = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if back_and_forth:
        M2 = cv2.getRotationMatrix2D(c, -theta_deg, 1.0)
        rot = cv2.warpAffine(rot, M2, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    m = window_mask(x0, y0)[..., None]
    return np.clip(m * rot.astype(np.float32) + (1 - m) * img.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
