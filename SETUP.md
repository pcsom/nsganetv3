# Environment Setup

## Install (One-time)

```bash
bash setup_environment.sh
```

Or manually:

```bash
conda create -n nsganetv2-llm python=3.10
conda activate nsganetv2-llm
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.6.13 pymoo==0.6.1.5 torchprofile gdown scipy
git clone https://github.com/mit-han-lab/once-for-all.git
cd once-for-all && pip install -e .
```

## Download OFA Checkpoint

The OFA MobileNetV3 supernet checkpoint is required:

```bash
mkdir -p checkpoints
cd checkpoints
gdown 1qmq7vWW6QkOPHfqXNnVvYQCFCQ2P6PUP
cd ..
```

Or download from: https://github.com/mit-han-lab/once-for-all/releases/download/v0.1-ckpts/ofa_mbv3_d234_e346_k357_w1.0

Place the checkpoint file at: `checkpoints/ofa_mbv3_d234_e346_k357_w1.0`

## Verify

```bash
bash verify_setup.sh
```