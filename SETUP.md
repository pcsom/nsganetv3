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

## Verify

```bash
bash verify_setup.sh
```