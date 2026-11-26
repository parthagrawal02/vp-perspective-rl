NCSA Delta & DeltaAI Quickstart Guide
NCSA runs two GPU clusters, and our course has an allocation on each:

Delta: x86 CPUs with NVIDIA A40, A100, and H200 GPUs. Course account: bjop-delta-gpu
DeltaAI: NVIDIA GH200 Grace Hopper nodes (an H100-class GPU with 96 GB of memory, paired with an ARM CPU). Course account: bjop-dtai-gh

They are separate machines. Each has its own login nodes, home directory, and software, so code you set up on one isn't available on the other. Most of this guide applies to both. Wherever they differ, there's an On DeltaAI note.
Account Setup
So by this point, you’ve already created an Access CI username/password. Now you need access to NCSA.

Your NCSA username is available from Access CI https://account.access-ci.org/ssh-keys
You can set your NCSA password here https://identity.ncsa.illinois.edu/reset

Next it will prompt you to set up two-factor authentication. You can use Duo Mobile (the same app used by CMU for two-factor).

The same NCSA username, password, and Duo work on both Delta and DeltaAI.
Login
To log in, you should use ssh. Detailed instructions are here:
https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html#ssh-log-in-steps

But the gist is that you run the following, replacing YOUR_NCSA_USERNAME with your username.

ssh YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu
...
YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu's password: <ENTER YOUR PASSWORD HERE>
...
Passcode or option (1-1): <ENTER 1 HERE TO GET A PUSH TO DUO ON YOUR PHONE>
...
[YOUR_NCSA_USERNAME@dt-login03 ~]$

Do NOT run your code on the login node. Instead, follow the instructions below on how to access a compute node.
On DeltaAI
Use the DeltaAI login host instead:

ssh YOUR_NCSA_USERNAME@dtai-login.delta.ncsa.illinois.edu

The prompt will show a gh-login node. Jobs for DeltaAI must be submitted from a DeltaAI login node, and jobs for Delta from a Delta login node.
Copying files
Copying files works the similarly:

scp my_program.py YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu:~/
rsync -av my_project/ YOUR_NCSA_USERNAME@dtai-login.delta.ncsa.illinois.edu:~/my_project/

Notes:

To close the shared connection early (e.g., if it gets stuck), run ssh -O exit delta.
Windows users: the built-in Windows SSH client doesn't support connection sharing. Use WSL (Windows Subsystem for Linux), where these instructions work as written. The Host shortcuts still work without the last block.
Where to Put Your Files
Location
Size
Shared by
Use it for
~ (home)
100 GB per student
Just you; Delta and DeltaAI each have a separate one
Your code, conda environments, small files
/work/hdd/bjop/$USER
Shared course quota
The whole class; the same files are visible from Delta and DeltaAI
Datasets, model checkpoints, Hugging Face downloads, job outputs
/projects/bjop
500 GB total
The whole class
Course-provided materials only. Please don't store your own files here.


Run quota on a login node to check your usage. Neither location is backed up, so keep your code in git.
Setting up your environment
Both systems use modules to provide software. You load a module to get conda, then create your own conda environment and install PyTorch into it. You only need to do this once per system.

Where: Creating the environment and installing packages on the login node is fine (it's light work and needs internet access). Testing that PyTorch can see a GPU must happen on a compute node (see below).

On Delta:

Step 1: Move caches out of your home directory. Hugging Face, pip, and PyTorch download models and data into ~/.cache by default, which will quickly fill your home quota. Point them at /work instead:

mkdir -p /work/hdd/bjop/$USER
cat >> ~/.bashrc << 'EOF'

# Keep large caches (Hugging Face models/datasets, pip, torch) out of $HOME
export XDG_CACHE_HOME=/work/hdd/bjop/$USER/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
EOF
source ~/.bashrc

Step 2: Create your environment and install PyTorch. It's often good to avoid the newest Python release, since PyTorch support lags behind slightly.

module load pytorch-conda     # provides conda; run `module avail conda` to see versions
conda create -p ~/envs/delta-env python=3.12 -y
conda activate ~/envs/delta-env
pip install torch torchvision       # Linux wheels from PyPI include CUDA support
pip install numpy matplotlib        # ...plus whatever else you need

Step 3: Activate automatically on every login. This adds the activation commands to your ~/.bashrc, so your environment is active whenever you log in or start an interactive job:

cat >> ~/.bashrc << 'EOF'

# Activate my PyTorch environment (interactive shells only)
if [[ $- == *i* ]]; then
    module load pytorch-conda
    conda activate ~/envs/delta-env
fi
EOF
source ~/.bashrc

Your prompt should now start with the environment's path in parentheses. The if line keeps activation from running for scp, rsync, and VS Code connections, where loading modules could slow them down or break them.
On DeltaAI
DeltaAI has ARM (aarch64) CPUs and a separate home directory, so you repeat the setup there with a different module and PyTorch install command:

# Step 1: caches. This is the same /work path as on Delta, so models you download
# on one system are already cached on the other.
mkdir -p /work/hdd/bjop/$USER
cat >> ~/.bashrc << 'EOF'

# Keep large caches (Hugging Face models/datasets, pip, torch) out of $HOME
export XDG_CACHE_HOME=/work/hdd/bjop/$USER/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
EOF
source ~/.bashrc

# Step 2: environment
module load python/miniforge3_pytorch
conda create -p ~/envs/dtai-env python=3.12 -y
conda activate ~/envs/dtai-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy matplotlib   # ...plus whatever else you need

# Step 3: automatic activation
cat >> ~/.bashrc << 'EOF'

# Activate my PyTorch environment (interactive shells only)
if [[ $- == *i* ]]; then
    module load python/miniforge3_pytorch
    conda activate ~/envs/dtai-env
fi
EOF
source ~/.bashrc

Tips (both systems):

Don't use pip install --user. It installs into ~/.local, which can conflict with your environment.
If you ever need a clean shell without the environment (e.g., to create a second environment), run conda deactivate.
If your home directory fills up, conda clean -a removes conda's cached package downloads.

To check that everything works, start an interactive GPU job (next section) and run:

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

nvidia-smi shows the GPU you were given. The Python line should print True and the GPU's name.
Compute Nodes / Submitting Jobs
To run your code, you must first get access to a compute node. You can do this either with (1) an interactive job or (2) a batch (asynchronous) job.

All jobs charge one of the course allocations, which have a limited budget shared by the whole class. Run accounts on each system to see how much is left. Please be economical: debug on small inputs first, exit interactive sessions when you're done, and cancel jobs you don't need.
What GPUs are Available?
You're billed in GPU-hours. Each GPU you request comes with a fixed share of CPUs and RAM. Requesting more CPUs or RAM than that share bills you for extra GPUs, even if you don't use them. So always request exactly one share per GPU, as in the table:

System
GPU
GPU memory
Batch partition
Interactive partition
CPUs per GPU
RAM per GPU
Cost (batch / interactive)
Delta
NVIDIA A40
48 GB
gpuA40x4
gpuA40x4-interactive
16
62G
0.5× / 1×
Delta
NVIDIA A100
40 GB
gpuA100x4
gpuA100x4-interactive
16
62G
1× / 2×
Delta
NVIDIA A100 (8-GPU nodes)
40 GB
gpuA100x8
gpuA100x8-interactive
16
248G
1.5× / 3×
Delta
NVIDIA H200
141 GB
gpuH200x8
gpuH200x8-interactive
12
248G
3× / 6×
DeltaAI
NVIDIA GH200
96 GB
ghx4
ghx4-interactive
72
100G
1× / 2×


Notes:

Which to use: Start with the A40 on Delta. It's the cheapest option. Use the GH200 on DeltaAI when you need more speed or GPU memory: it's much faster than an A100 at the same price, but remember that DeltaAI needs its own environment. Use the H200 only when you need its 141 GB of GPU memory.
Interactive partitions cost 2×. They're limited to 1 hour on Delta and 2 hours on DeltaAI, and to one job at a time per user. Batch partitions allow up to 48 hours.
Delta also has AMD MI100 GPUs (gpuMI100x8). Standard PyTorch (CUDA) doesn't run on them, so they're left out here.
Interactive Jobs
An interactive job gives you a shell on a compute node with a GPU, which is good for debugging and short experiments.

On Delta:

srun -A bjop-delta-gpu -p gpuA40x4-interactive \
  --nodes=1 --gpus-per-node=1 --ntasks=1 \
  --cpus-per-task=16 --mem=62G --time=01:00:00 \
  --pty bash

Once the job starts, your prompt changes to the compute node's name (e.g., gpub012), and your environment is already active. Just run your code:

python my_program.py

Type exit when you're done, so you stop being charged.

To use a different Delta GPU, change three flags using the table above:

-p → that GPU's interactive partition (e.g., gpuA100x4-interactive)
--cpus-per-task → its CPUs per GPU
--mem → its RAM per GPU

For more than one GPU, multiply all three: e.g., 2 A100s is --gpus-per-node=2 --cpus-per-task=32 --mem=124G.
On DeltaAI
From a DeltaAI login node:

srun -A bjop-dtai-gh -p ghx4-interactive \
  --nodes=1 --gpus-per-node=1 --ntasks=1 \
  --cpus-per-task=72 --mem=100G --time=02:00:00 \
  --pty bash

Your DeltaAI environment activates automatically. Run your code as above.
Batch Jobs
A batch job runs your code in the background, without you watching. You can log out and come back later. This is how you should run anything longer than a few minutes.

Your environment is already active when you log in, and the batch job inherits it (along with the cache settings), so you can submit directly. (If you ever run conda deactivate or switch environments, re-activate before submitting.)

On Delta:

sbatch -A bjop-delta-gpu -p gpuA40x4 \
  --nodes=1 --gpus-per-node=1 --ntasks=1 \
  --cpus-per-task=16 --mem=62G --time=04:00:00 \
  -o slurm-%j.out \
  --wrap="python my_program.py"

Equivalently, you can put the options in a script, job.slurm:

#!/bin/bash
#SBATCH --account=bjop-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

python my_program.py

and submit it with sbatch job.slurm. Any flag you give on the command line overrides the matching #SBATCH line, e.g., sbatch --time=12:00:00 job.slurm.

To use a different Delta GPU, change the same three things as for interactive jobs, using the table above:

-p → that GPU's batch partition (e.g., gpuA100x4)
--cpus-per-task → its CPUs per GPU
--mem → its RAM per GPU
On DeltaAI
From a DeltaAI login node:

sbatch -A bjop-dtai-gh -p ghx4 \
  --nodes=1 --gpus-per-node=1 --ntasks=1 \
  --cpus-per-task=72 --mem=100G --time=04:00:00 \
  -o slurm-%j.out \
  --wrap="python my_program.py"

In a job.slurm script, change --account, --partition, --cpus-per-task, and --mem to these values.
Useful Commands (both systems)
squeue -u $USER shows your jobs and their status (PD = waiting, R = running).
scancel <jobid> cancels a job.
cat slurm-<jobid>.out shows your program's output.
nvidia-smi (on a compute node) shows the GPU's memory use and utilization. watch -n 2 nvidia-smi refreshes it every 2 seconds.
accounts shows the remaining budget for the course allocations.
quota shows your disk usage.

Always set --time to a realistic limit. Your job is killed when the limit runs out. The default is only 30 minutes, and shorter jobs often start sooner.


