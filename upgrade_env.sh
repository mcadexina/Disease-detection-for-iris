#!/bin/bash
# Upgrade local environment to NumPy 2.x with TensorFlow 2.17+

echo "========================================"
echo "Upgrading Local Environment"
echo "========================================"
echo ""

# Activate virtual environment
echo "[1/5] Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate venv. Make sure venv exists."
    exit 1
fi

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
python -m pip install --upgrade pip

# Uninstall old conflicting packages
echo ""
echo "[3/5] Uninstalling old packages..."
pip uninstall -y tensorflow tensorflow-cpu keras numpy scikit-learn scikit-image

# Install new requirements
echo ""
echo "[4/5] Installing new requirements..."
pip install -r requirements.txt

# Test compatibility
echo ""
echo "[5/5] Testing model compatibility..."
python test_model_compatibility.py

echo ""
echo "========================================"
echo "Environment upgrade complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  If models failed to load, retrain them:"
echo "  python train_disease_models.py --models all"
echo ""
