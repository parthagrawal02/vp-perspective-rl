#!/bin/bash
# Task 12: nft-vp = exact copy of the nft env (conda + pip packages) + DeepLSD deps. micromamba --clone copies conda packages
# only (torch etc. are pip-installed), so the env directory is copied and script shebangs rewritten.
set -e
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh
S=$VP_DATA/envs/nft; E=$VP_DATA/envs/nft-vp
[ -x $E/bin/python ] || cp -a $S $E
grep -rlI "^#!$S/bin/" $E/bin | xargs -r sed -i "1s|^#!$S/bin/|#!$E/bin/|"   # idempotent shebang rewrite
# --no-deps: their only heavy dep (torch) is already installed; pip cannot resolve the local torch==2.7.1+cu128 pin from the index
$E/bin/python -m pip install --no-deps omegaconf==2.4.0 PyYAML==6.0.3 kornia==0.8.2 kornia_rs==0.2.0
rm -rf $VP_DATA/tmp/pytlsd_src && mkdir -p $VP_DATA/tmp && rsync -a --exclude build --exclude '*.egg-info' $VP_DEEPLSD_DIR/third_party/pytlsd/ $VP_DATA/tmp/pytlsd_src/
# pytlsd 0.0.5 is not on PyPI; built from source (out-of-tree copy); CMake 4 needs the policy floor for the bundled pybind11
CMAKE_POLICY_VERSION_MINIMUM=3.5 $E/bin/python -m pip install --no-deps $VP_DATA/tmp/pytlsd_src
$E/bin/python -m pip install --no-deps -e $VP_DEEPLSD_DIR
$E/bin/python -m pip list 2>/dev/null | grep -i -E "^(torch|torchvision|diffusers|transformers|peft|tokenizers|huggingface-hub|omegaconf|kornia|pytlsd|deeplsd|opencv-python|numpy|pyyaml) "
diff <($S/bin/python -m pip freeze 2>/dev/null | grep -i -E "^(torch|torchvision|diffusers|transformers|peft|tokenizers|huggingface-hub)=") <($E/bin/python -m pip freeze 2>/dev/null | grep -i -E "^(torch|torchvision|diffusers|transformers|peft|tokenizers|huggingface-hub)=") && echo CORE_PINS_IDENTICAL
$E/bin/python -c "import torch, deeplsd, omegaconf, kornia, pytlsd; from deeplsd.models.deeplsd_inference import DeepLSD; print('IMPORT_OK', torch.__version__, torch.cuda.get_arch_list()[-2:])"
echo ENV_DONE
