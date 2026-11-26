# NCSA Delta settings, layered on config/env.example.sh. rl/delta/setup.sh copies this to config/env.sh (git-ignored).
# Big files go to the shared course quota on /work (not backed up); the conda env lives in $HOME as the NCSA guide recommends.
export VP_DATA="${VP_DATA:-/work/hdd/bjop/$USER/vp-data}"
export VP_ENV="${VP_ENV:-$HOME/envs/nft-vp}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.example.sh"
