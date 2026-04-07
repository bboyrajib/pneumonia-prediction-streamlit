#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Automated Pneumonia and COVID-19 Detection — Local venv Setup"
echo " PyTorch CUDA 12.4  |  RTX 3060  |  Python 3.11+"
echo "============================================================"
echo

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python not found. Install Python 3.11+ and add it to PATH."
    exit 1
fi
echo "Found: $(python3 --version)"

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "[skip] .venv already exists"
else
    echo "[1/5] Creating virtual environment in .venv ..."
    python3 -m venv .venv
    echo "      Done."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
echo "[2/5] Activating .venv ..."
source .venv/bin/activate

# ── Upgrade pip ───────────────────────────────────────────────────────────────
echo "[3/5] Upgrading pip ..."
python -m pip install --upgrade pip --quiet

# ── Install PyTorch (CUDA 12.4 wheel) ─────────────────────────────────────────
echo "[4/5] Installing PyTorch with CUDA 12.4 support ..."
echo "      (This may take several minutes — ~2 GB download)"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet
echo "      PyTorch installed."

# ── Install remaining requirements ────────────────────────────────────────────
echo "[5/5] Installing project dependencies ..."
pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    albumentations==1.4.3 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    streamlit==1.35.0 \
    plotly==5.22.0 \
    kaggle==1.6.14 \
    jupyter==1.0.0 \
    ipykernel==6.29.4 \
    ipywidgets==8.1.3 \
    tqdm==4.66.4 \
    --quiet

# ── Register Jupyter kernel ───────────────────────────────────────────────────
echo "Registering Jupyter kernel \"pneumonia-detection\" ..."
python -m ipykernel install --user --name=pneumonia-detection --display-name "Pneumonia Detection (PyTorch)"
echo

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "── GPU Verification ─────────────────────────────────────────"
python verify_gpu.py
echo

echo "============================================================"
echo " Setup complete!"
echo
echo " To train via Jupyter notebook:"
echo "   source .venv/bin/activate"
echo "   jupyter notebook notebooks/Automated_Pneumonia_Detection_from_Chest_X_Ray_Images_pytorch.ipynb"
echo
echo " To train via Python script directly:"
echo "   source .venv/bin/activate"
echo "   python notebooks/automated_pneumonia_detection_from_chest_x_ray_images_pytorch.py"
echo
echo " To run the Streamlit app:"
echo "   source .venv/bin/activate"
echo "   streamlit run app.py"
echo "============================================================"
