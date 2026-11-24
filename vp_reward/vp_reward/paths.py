"""Single source of paths. Everything is configured by environment variables (see config/env.example.sh); defaults:
  VP_DATA          ~/vp-data                         datasets/, outputs/, models/, checkpoints/ (never inside the repo)
  VP_THIRD_PARTY   <repo>/third_party                upstream clones
  VP_NFT_DIR       $VP_THIRD_PARTY/DiffusionNFT      NVlabs/DiffusionNFT @ cbb14f8 + rl/diffusionnft_vp.patch
  VP_DEEPLSD_DIR   $VP_THIRD_PARTY/DeepLSD           cvg/DeepLSD @ f7d9d62
  VP_DEEPLSD_CKPT  $VP_DATA/models/deeplsd/deeplsd_wireframe.tar
REPO assumes the editable install (`pip install -e vp_reward`) of this repo."""
import os
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
DATA = os.path.expanduser(os.environ.get("VP_DATA", "~/vp-data"))
THIRD_PARTY = os.environ.get("VP_THIRD_PARTY", os.path.join(REPO, "third_party"))
NFT_DIR = os.environ.get("VP_NFT_DIR", os.path.join(THIRD_PARTY, "DiffusionNFT"))
DEEPLSD_DIR = os.environ.get("VP_DEEPLSD_DIR", os.path.join(THIRD_PARTY, "DeepLSD"))
DEEPLSD_CKPT = os.environ.get("VP_DEEPLSD_CKPT", os.path.join(DATA, "models", "deeplsd", "deeplsd_wireframe.tar"))


def src(name):
    """Directory of a code component by its old (pre-repo) name: 'vp_reward', 'DiffusionNFT', 'DeepLSD', 'task3' -> experiments/task03, ..."""
    if name == "vp_reward":
        return os.path.join(REPO, "vp_reward")
    if name == "DiffusionNFT":
        return NFT_DIR
    if name == "DeepLSD":
        return DEEPLSD_DIR
    if name.startswith("task"):
        return os.path.join(REPO, "experiments", f"task{int(name[4:]):02d}")
    if name == "tools":
        return os.path.join(REPO, "experiments", "task01")
    raise KeyError(name)


def add_nft_to_path():
    """flow_grpo (PickScore / aesthetic scorers) lives in the DiffusionNFT clone and is not an installable package."""
    import sys
    if NFT_DIR not in sys.path:
        sys.path.insert(0, NFT_DIR)
