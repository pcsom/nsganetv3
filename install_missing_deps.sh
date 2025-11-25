#!/bin/bash
# Install missing dependencies for NSGANetV3
# Usage: bash install_missing_deps.sh

set -e

echo "======================================"
echo "Installing Missing NSGANetV3 Dependencies"
echo "======================================"
echo ""

# Load modules
echo "Loading modules..."
module load anaconda3/2023.03

# Activate environment
echo "Activating conda environment 'nas'..."
eval "$(conda shell.bash hook)"
conda activate nas

# Check if environment exists
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate 'nas' environment."
    echo "Please run: bash setup_environment.sh first"
    exit 1
fi

echo ""
echo "Installing missing packages..."
echo ""

# Install torchprofile
echo "[1/5] Installing torchprofile..."
pip install torchprofile==0.0.1

# Install timm
echo "[2/5] Installing timm..."
pip install timm==0.1.30

# Install OFA
echo "[3/5] Installing OnceForAll (ofa)..."
pip install ofa==0.0.4

# Install pySOT
echo "[4/5] Installing pySOT..."
pip install pySOT==0.2.3

# Install pydacefit
echo "[5/5] Installing pydacefit..."
pip install pydacefit==1.0.1

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "
import sys
packages = ['pymoo', 'torchprofile', 'timm', 'ofa', 'pySOT', 'pydacefit', 'toml']
all_ok = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - FAILED')
        all_ok = False
if all_ok:
    print('\n✓ All dependencies installed successfully!')
else:
    print('\n✗ Some dependencies failed to install')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "You're ready to run NSGANetV3!"
    echo "Next: bash quick_start.sh"
else
    echo ""
    echo "Some packages failed to install. Please check the errors above."
    exit 1
fi
