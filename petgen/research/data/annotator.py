"""Semi-automatic annotation tool using SAM2 for mouth segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import Landmark

logger = logging.getLogger(__name__)


class SAM2Annotator:
    """Semi-automatic annotation tool using SAM2 for mouth segmentation."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._sam2_predictor = None
        self._sam2_video_predictor = None

    def _load_sam2(self):
        """Lazy-load SAM2 image predictor."""
        if self._sam2_predictor is not None:
            return self._sam2_predictor

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = self.config.weights_dir / "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        sam2_model = build_sam2(model_cfg, str(checkpoint), device=self.config.device)
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        return self._sam2_predictor

    def _load_sam2_video(self):
        """Lazy-load SAM2 video predictor."""
        if self._sam2_video_predictor is not None:
            return self._sam2_video_predictor

        from sam2.build_sam import build_sam2_video_predictor

        checkpoint = self.config.weights_dir / "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self._sam2_video_predictor = build_sam2_video_predictor(
            model_cfg, str(checkpoint), device=self.config.device,
        )
        return self._sam2_video_predictor

    def seed_from_landmarks(
        self, frame: np.ndarray, landmarks: list[Landmark]
    ) -> np.ndarray:
        """Generate initial mouth mask from landmark points as SAM2 prompts.

        Uses mouth landmark coordinates as positive point prompts for SAM2
        to produce an initial segmentation mask.

        Args:
            frame: RGB image array (H, W, 3).
            landmarks: Mouth landmark points to use as prompts.

        Returns:
            Binary mask (H, W) with uint8 values 0 or 255.
        """
        if not landmarks:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        predictor = self._load_sam2()
        predictor.set_image(frame)

        # Use landmark points as positive prompts
        point_coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        point_labels = np.ones(len(landmarks), dtype=np.int32)  # all positive

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick the mask with the highest score
        best_idx = int(np.argmax(scores))
        mask = (masks[best_idx] > 0.5).astype(np.uint8) * 255
        return mask

    def propagate_masks(
        self, frames: list[np.ndarray], initial_mask: np.ndarray
    ) -> list[np.ndarray]:
        """Use SAM2 Video Predictor to propagate mask across frames.

        Args:
            frames: List of RGB frame arrays.
            initial_mask: Binary mask for the first frame (H, W), uint8.

        Returns:
            List of binary masks, one per frame.
        """
        if not frames:
            return []

        if initial_mask.max() == 0:
            return [np.zeros(f.shape[:2], dtype=np.uint8) for f in frames]

        video_predictor = self._load_sam2_video()
        state = video_predictor.init_state(frames=frames)

        # Add initial mask as prompt on frame 0
        # Convert mask to point prompts (centroid + border points)
        ys, xs = np.where(initial_mask > 0)
        if len(xs) == 0:
            return [np.zeros(f.shape[:2], dtype=np.uint8) for f in frames]

        # Use centroid as the prompt point
        cx, cy = float(xs.mean()), float(ys.mean())
        video_predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            points=np.array([[cx, cy]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        masks_out = [np.zeros(f.shape[:2], dtype=np.uint8) for f in frames]
        masks_out[0] = initial_mask

        for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            masks_out[frame_idx] = mask

        return masks_out

    def refine_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
    ) -> np.ndarray:
        """Refine mask with additional click points.

        Args:
            frame: RGB image (H, W, 3).
            mask: Current binary mask (H, W).
            positive_points: Points that should be inside the mask.
            negative_points: Points that should be outside the mask.

        Returns:
            Refined binary mask (H, W), uint8.
        """
        predictor = self._load_sam2()
        predictor.set_image(frame)

        all_points = list(positive_points) + list(negative_points)
        labels = [1] * len(positive_points) + [0] * len(negative_points)

        if not all_points:
            return mask

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask[None, :, :].astype(np.float32) if mask.max() > 0 else None,
            multimask_output=False,
        )

        return (masks[0] > 0.5).astype(np.uint8) * 255

    def export_annotations(
        self, frames_dir: Path, masks: list[np.ndarray], output_dir: Path
    ) -> Path:
        """Export frame-mask pairs as a training dataset.

        Creates output_dir with:
        - frames/ — symlinks or copies of original frames
        - masks/ — corresponding mask images
        - pairs.json — mapping of frame to mask paths

        Args:
            frames_dir: Directory containing source frames.
            masks: List of mask arrays corresponding to sorted frames.
            output_dir: Where to write the exported dataset.

        Returns:
            Path to the pairs.json manifest.
        """
        import cv2
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        out_frames = output_dir / "frames"
        out_masks = output_dir / "masks"
        out_frames.mkdir(exist_ok=True)
        out_masks.mkdir(exist_ok=True)

        frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        pairs = []

        for i, (fp, mask) in enumerate(zip(frame_paths, masks)):
            frame_dest = out_frames / fp.name
            mask_name = f"mask_{fp.stem}.png"
            mask_dest = out_masks / mask_name

            # Copy frame
            import shutil
            shutil.copy2(fp, frame_dest)

            # Save mask
            cv2.imwrite(str(mask_dest), mask)

            pairs.append({
                "frame": str(frame_dest),
                "mask": str(mask_dest),
                "index": i,
            })

        manifest_path = output_dir / "pairs.json"
        manifest_path.write_text(json.dumps(pairs, indent=2))
        logger.info("Exported %d frame-mask pairs to %s", len(pairs), output_dir)
        return manifest_path
