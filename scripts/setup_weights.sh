#!/usr/bin/env bash
# PetGen — Download all model weights for GPU inference
# Run this on a machine with GPU (Colab, RunPod, etc.)
set -euo pipefail

WEIGHTS_DIR="${PETGEN_WEIGHTS_DIR:-$HOME/.petgen/weights}"
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "=== PetGen Model Weight Setup ==="
echo "Target directory: $WEIGHTS_DIR"
echo ""

# --- 1. JoyVASA (includes LivePortrait animal weights) ---
echo "[1/4] Downloading JoyVASA + LivePortrait weights from HuggingFace..."
if [ ! -d "$WEIGHTS_DIR/joyvasa" ]; then
    huggingface-cli download jdh-algo/JoyVASA --local-dir "$WEIGHTS_DIR/joyvasa_repo"
    # Organize into expected structure
    ln -sfn "$WEIGHTS_DIR/joyvasa_repo" "$WEIGHTS_DIR/joyvasa"
    echo "  ✓ JoyVASA weights downloaded"
else
    echo "  ✓ JoyVASA weights already present"
fi

# --- 2. LivePortrait animal weights (from KlingTeam) ---
echo "[2/4] Downloading LivePortrait animal weights..."
if [ ! -d "$WEIGHTS_DIR/liveportrait_animal" ]; then
    huggingface-cli download KlingTeam/LivePortrait --local-dir "$WEIGHTS_DIR/liveportrait_all" \
        --exclude "*.git*" "README.md" "docs"
    ln -sfn "$WEIGHTS_DIR/liveportrait_all/liveportrait_animals" "$WEIGHTS_DIR/liveportrait_animal"
    echo "  ✓ LivePortrait animal weights downloaded"
else
    echo "  ✓ LivePortrait animal weights already present"
fi

# --- 3. SAM2 (segment-anything-2 small checkpoint) ---
echo "[3/4] Downloading SAM2 checkpoint..."
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
if [ ! -f "$WEIGHTS_DIR/sam2_hiera_small.pt" ]; then
    wget -q --show-progress -O "$WEIGHTS_DIR/sam2_hiera_small.pt" "$SAM2_URL"
    echo "  ✓ SAM2 hiera small downloaded"
else
    echo "  ✓ SAM2 checkpoint already present"
fi

# --- 4. YOLOv8 (auto-downloads on first use, but pre-cache it) ---
echo "[4/4] Pre-caching YOLOv8n..."
if [ ! -f "$WEIGHTS_DIR/yolov8n.pt" ]; then
    python3 -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); import shutil; shutil.move('yolov8n.pt', '$WEIGHTS_DIR/yolov8n.pt')" 2>/dev/null || \
    wget -q --show-progress -O "$WEIGHTS_DIR/yolov8n.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    echo "  ✓ YOLOv8n downloaded"
else
    echo "  ✓ YOLOv8n already present"
fi

echo ""
echo "=== Chatterbox TTS & wav2vec2 ==="
echo "These auto-download from HuggingFace on first inference run."
echo "No manual download needed."
echo ""

echo "=== All weights ready ==="
echo "Directory listing:"
ls -lh "$WEIGHTS_DIR/"
echo ""
echo "Set PETGEN_WEIGHTS_DIR=$WEIGHTS_DIR in your environment."
