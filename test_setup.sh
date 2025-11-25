#!/bin/bash
# Test script to verify NSGANetV3 setup on PACE ICE
# Usage: bash test_setup.sh

echo "======================================"
echo "NSGANetV3 Setup Verification"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((FAILED++))
    fi
}

# Test 1: Check if we're on PACE ICE
echo "Test 1: Checking PACE ICE environment..."
if [[ $(hostname) == *"ice"* ]]; then
    print_result 0 "Running on PACE ICE cluster"
else
    print_result 1 "Not running on PACE ICE (hostname: $(hostname))"
fi

# Test 2: Check module system
echo ""
echo "Test 2: Checking module availability..."
if command -v module &> /dev/null; then
    print_result 0 "Module command available"
else
    print_result 1 "Module command not found"
fi

# Test 3: Check anaconda module
echo ""
echo "Test 3: Checking anaconda module..."
module load anaconda3/2023.03 &> /dev/null
if [ $? -eq 0 ]; then
    print_result 0 "Anaconda module loaded successfully"
else
    print_result 1 "Failed to load anaconda module"
fi

# Test 4: Check CUDA module
echo ""
echo "Test 4: Checking CUDA module..."
module load cuda/12.1.1 &> /dev/null
if [ $? -eq 0 ]; then
    print_result 0 "CUDA module loaded successfully"
else
    print_result 1 "Failed to load CUDA module"
fi

# Test 5: Check conda environment
echo ""
echo "Test 5: Checking conda environment 'nas'..."
if conda env list | grep -q "^nas "; then
    print_result 0 "Conda environment 'nas' exists"
    
    # Test 5b: Check if PyTorch is installed
    echo ""
    echo "Test 5b: Checking PyTorch installation..."
    eval "$(conda shell.bash hook)"
    conda activate nas 2>/dev/null
    python -c "import torch; print('PyTorch version:', torch.__version__)" &> /tmp/pytorch_test.txt
    if [ $? -eq 0 ]; then
        print_result 0 "PyTorch installed ($(cat /tmp/pytorch_test.txt))"
    else
        print_result 1 "PyTorch not installed or failed to import"
    fi
    
    # Test 5c: Check other dependencies
    echo ""
    echo "Test 5c: Checking other dependencies..."
    python -c "import pymoo, torchprofile, timm, ofa" &> /dev/null
    if [ $? -eq 0 ]; then
        print_result 0 "Core dependencies installed (pymoo, torchprofile, timm, ofa)"
    else
        print_result 1 "Some dependencies missing"
    fi
    
    conda deactivate 2>/dev/null
else
    print_result 1 "Conda environment 'nas' not found - run setup_environment.sh"
fi

# Test 6: Check project files
echo ""
echo "Test 6: Checking project files..."
if [ -f "/home/hice1/glu49/nsganetv3/msunas_slurm.py" ]; then
    print_result 0 "Main Python script found"
else
    print_result 1 "Main Python script not found"
fi

if [ -f "/home/hice1/glu49/nsganetv3/run_nsganetv3_slurm.sh" ]; then
    print_result 0 "SLURM batch script found"
else
    print_result 1 "SLURM batch script not found"
fi

if [ -f "/home/hice1/glu49/nsganetv3/config/nsganetv3_config.toml" ]; then
    print_result 0 "Configuration file found"
else
    print_result 1 "Configuration file not found"
fi

# Test 7: Check storage directories
echo ""
echo "Test 7: Checking storage directories..."
if [ -d "/storage/ice-shared/vip-vvk/data/" ]; then
    print_result 0 "Storage directory accessible"
else
    print_result 1 "Storage directory not accessible"
fi

# Test 8: Check SLURM commands
echo ""
echo "Test 8: Checking SLURM commands..."
if command -v sbatch &> /dev/null; then
    print_result 0 "sbatch command available"
else
    print_result 1 "sbatch command not found"
fi

if command -v squeue &> /dev/null; then
    print_result 0 "squeue command available"
else
    print_result 1 "squeue command not found"
fi

# Summary
echo ""
echo "======================================"
echo "Summary"
echo "======================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! You're ready to run NSGANetV3.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Download supernet weights (see SETUP_PACE_ICE.md)"
    echo "2. Prepare your dataset"
    echo "3. Run: bash quick_start.sh"
else
    echo -e "${YELLOW}Some tests failed. Please address the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    if ! conda env list | grep -q "^nas "; then
        echo "- Run: bash setup_environment.sh"
    fi
    echo "- Check SETUP_PACE_ICE.md for detailed setup instructions"
fi

echo ""
