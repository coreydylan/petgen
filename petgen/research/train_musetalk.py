"""Training script for MuseTalk animal mouth model.

Usage:
    python -m petgen.research.train_musetalk \
        --data_dir /path/to/dataset \
        --output_dir /path/to/checkpoints \
        --epochs 100 \
        --batch_size 8 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AnimalMouthDataset:
    """Dataset for animal mouth training data.

    Loads frame-mask-audio triplets from a prepared dataset directory.
    Compatible with PyTorch DataLoader when torch is available.

    Expected directory structure:
        data_dir/
            pairs.json         # frame-mask mapping
            frames/            # RGB frame images
            masks/             # binary mouth masks
            audio_features/    # pre-extracted wav2vec2 features (.npy)
    """

    def __init__(self, data_dir: Path, transform=None) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self._pairs: list[dict] = []

        pairs_path = data_dir / "pairs.json"
        if pairs_path.exists():
            self._pairs = json.loads(pairs_path.read_text())

        self._audio_features = None
        audio_dir = data_dir / "audio_features"
        if audio_dir.exists():
            feature_files = sorted(audio_dir.glob("*.npy"))
            if feature_files:
                self._audio_features = [np.load(f) for f in feature_files]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        """Load a single training sample.

        Returns:
            Dict with keys: frame, mouth_mask, audio_features, target_frame.
            All values are numpy arrays. frame and target_frame are (H, W, 3)
            float32 [0, 1]. mouth_mask is (H, W) float32 [0, 1].
            audio_features is (D,) float32.
        """
        import cv2

        pair = self._pairs[idx]
        frame_path = pair["frame"]
        mask_path = pair["mask"]

        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            frame = np.zeros((256, 256, 3), dtype=np.float32)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.zeros((256, 256), dtype=np.float32)

        # Audio features
        audio_feat = np.zeros(768, dtype=np.float32)
        if self._audio_features is not None and idx < len(self._audio_features):
            audio_feat = self._audio_features[idx].astype(np.float32)

        # For training, target_frame is the same frame (self-supervised reconstruction)
        target_frame = frame.copy()

        sample = {
            "frame": frame,
            "mouth_mask": mask,
            "audio_features": audio_feat,
            "target_frame": target_frame,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class MouthInpaintingLoss:
    """Combined loss for mouth inpainting training.

    Combines L1 reconstruction loss, perceptual loss, and adversarial loss,
    weighted by configurable factors. All computations use numpy for
    CPU-compatibility; convert to torch tensors for GPU training.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        adversarial_weight: float = 0.1,
    ) -> None:
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight

    def __call__(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> dict[str, float]:
        """Compute combined loss.

        Args:
            predicted: Predicted frame (H, W, 3) or (B, H, W, 3), float32.
            target: Target frame, same shape as predicted.
            mouth_mask: Binary mouth mask (H, W) or (B, H, W), float32.

        Returns:
            Dict with keys: total, l1, perceptual, adversarial.
        """
        # Ensure mouth_mask has channel dimension for broadcasting
        if mouth_mask.ndim == 2:
            mask = mouth_mask[:, :, np.newaxis]
        elif mouth_mask.ndim == 3 and mouth_mask.shape[-1] != 1:
            mask = mouth_mask[:, :, :, np.newaxis]
        else:
            mask = mouth_mask

        # L1 loss (focused on mouth region)
        diff = np.abs(predicted - target)
        if mask.sum() > 0:
            l1_loss = float((diff * mask).sum() / max(mask.sum(), 1.0))
        else:
            l1_loss = float(diff.mean())

        # Perceptual loss stub — in real training, use VGG features
        # Here we approximate with a simple MSE on 4x downsampled patches
        try:
            from skimage.transform import resize as sk_resize

            pred_small = sk_resize(predicted, (64, 64), anti_aliasing=True)
            tgt_small = sk_resize(target, (64, 64), anti_aliasing=True)
            perceptual_loss = float(np.mean((pred_small - tgt_small) ** 2))
        except ImportError:
            perceptual_loss = float(np.mean((predicted - target) ** 2))

        # Adversarial loss stub — returns 0 without a trained discriminator
        adversarial_loss = 0.0

        total = (
            self.l1_weight * l1_loss
            + self.perceptual_weight * perceptual_loss
            + self.adversarial_weight * adversarial_loss
        )

        return {
            "total": total,
            "l1": l1_loss,
            "perceptual": perceptual_loss,
            "adversarial": adversarial_loss,
        }


def train(args: argparse.Namespace) -> None:
    """Main training loop.

    Steps:
    1. Load dataset
    2. Create model (MuseTalkAnimalAdapter)
    3. Create optimizer (AdamW)
    4. Training loop:
       - Forward pass through audio encoder
       - Inpaint mouth region
       - Compute loss
       - Backward pass
       - Log metrics
       - Save checkpoints
    """
    logger.info("Starting training with args: %s", args)

    # Load dataset
    dataset = AnimalMouthDataset(args.data_dir)
    logger.info("Dataset loaded: %d samples", len(dataset))

    if len(dataset) == 0:
        logger.error("No training data found in %s", args.data_dir)
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create loss function
    loss_fn = MouthInpaintingLoss()

    # Training requires torch — check availability
    try:
        import torch
        import torch.optim as optim
    except ImportError:
        logger.error(
            "PyTorch is required for training. Install with: pip install torch"
        )
        return

    # Create model
    from petgen.config import PetGenConfig

    config = PetGenConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    from petgen.research.musetalk_adapter import MuseTalkAnimalAdapter

    model = MuseTalkAnimalAdapter(config)

    # NOTE: In a full implementation, the inpainter would be a trainable
    # nn.Module. This scaffold demonstrates the training loop structure.
    logger.info("Model created (scaffold mode — no trainable inpainter)")

    # Training loop scaffold
    for epoch in range(args.epochs):
        epoch_losses = []

        for i in range(len(dataset)):
            sample = dataset[i]
            predicted = sample["frame"]  # passthrough in scaffold mode
            target = sample["target_frame"]
            mask = sample["mouth_mask"]

            losses = loss_fn(predicted, target, mask)
            epoch_losses.append(losses["total"])

            if i % 100 == 0:
                logger.info(
                    "Epoch %d/%d, Step %d/%d, Loss: %.4f",
                    epoch + 1, args.epochs, i, len(dataset), losses["total"],
                )

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info("Epoch %d/%d complete, Avg Loss: %.4f", epoch + 1, args.epochs, avg_loss)

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch + 1}.json"
            ckpt_path.write_text(json.dumps({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "args": {
                    "data_dir": str(args.data_dir),
                    "output_dir": str(args.output_dir),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                },
            }, indent=2))
            logger.info("Checkpoint saved to %s", ckpt_path)

    logger.info("Training complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MuseTalk animal mouth model"
    )
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to training dataset directory")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to resume from checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
