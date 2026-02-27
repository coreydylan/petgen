"""Compositing and post-processing — artifact repair, fur matting, deflickering, encoding."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from petgen.config import PetGenConfig
from petgen.utils.video import frames_to_video, mux_audio_video, resize_frames

if TYPE_CHECKING:
    from petgen.models import BoundingBox
    from petgen.modules.inpainting import ProPainterInpainter
    from petgen.modules.matting import FurMatter

logger = logging.getLogger(__name__)


class Compositor:
    """Post-processing pipeline for final video output."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config

    def repair_artifacts(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fix warping artifacts around the animated mouth region.

        Prototype approach: Gaussian blur the mouth boundary region and alpha-blend
        it back for smooth transitions. Uses mouth_masks to identify the repair region.
        Later upgrade path: integrate ProPainter for dual-domain propagation.
        """
        if not frames or not mouth_masks:
            return list(frames)

        result = []
        for i, frame in enumerate(frames):
            if i >= len(mouth_masks):
                result.append(frame.copy())
                continue

            mask = mouth_masks[i]
            if mask.max() == 0:
                result.append(frame.copy())
                continue

            h, w = frame.shape[:2]

            # Ensure mask matches frame dimensions
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Normalize mask to [0, 1] float
            if mask.dtype == np.uint8:
                mask_f = mask.astype(np.float32) / 255.0
            else:
                mask_f = mask.astype(np.float32)

            # Ensure mask is 2D
            if mask_f.ndim == 3:
                mask_f = mask_f[:, :, 0]

            # Dilate the mask to capture the boundary region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(mask_f, kernel, iterations=2)

            # The boundary region is the dilated mask minus the original
            boundary = np.clip(dilated - mask_f * 0.5, 0, 1)

            # Gaussian blur the frame
            blurred = cv2.GaussianBlur(frame, (21, 21), 0)

            # Blend: use blurred pixels in the boundary, original elsewhere
            boundary_3ch = boundary[:, :, np.newaxis]
            repaired = (
                frame.astype(np.float32) * (1 - boundary_3ch)
                + blurred.astype(np.float32) * boundary_3ch
            ).astype(np.uint8)

            result.append(repaired)

        return result

    def blend_fur_boundaries(
        self,
        frames: list[np.ndarray],
        alpha_mattes: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Blend animated region with static background using alpha mattes.

        Simple alpha compositing: alpha * animated + (1-alpha) * original.
        Matte edges are feathered with Gaussian blur for smooth transitions.
        """
        if not frames or not alpha_mattes:
            return list(frames)

        # Use the first frame as the "static background" reference
        background = frames[0].copy()
        result = []

        for i, frame in enumerate(frames):
            if i >= len(alpha_mattes):
                result.append(frame.copy())
                continue

            matte = alpha_mattes[i]
            h, w = frame.shape[:2]

            # Ensure matte matches frame dimensions
            if matte.shape[:2] != (h, w):
                matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

            # Normalize to [0, 1] float
            if matte.dtype == np.uint8:
                alpha = matte.astype(np.float32) / 255.0
            else:
                alpha = matte.astype(np.float32)

            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]

            # Feather matte edges
            alpha_2d = alpha[:, :, 0]
            alpha_2d = cv2.GaussianBlur(alpha_2d, (11, 11), 0)
            alpha = alpha_2d[:, :, np.newaxis]

            # Alpha composite
            blended = (
                frame.astype(np.float32) * alpha
                + background.astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)

            result.append(blended)

        return result

    def generate_alpha_mattes(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Generate per-frame alpha mattes for fur boundary blending.

        Prototype approach: edge detection + dilation + Gaussian blur for soft matte.
        Later upgrade: integrate ViTMatte for production quality.
        """
        if not frames:
            return []

        mattes = []
        for frame in frames:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges to create a broader matte region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=3)

            # Fill the interior: flood fill from corners to find background,
            # then invert to get the subject matte
            filled = dilated.copy()
            h, w = filled.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, flood_mask, (0, 0), 255)
            filled_inv = cv2.bitwise_not(filled)

            # Combine with dilated edges
            matte = cv2.bitwise_or(dilated, filled_inv)

            # Gaussian blur for soft edges
            matte = cv2.GaussianBlur(matte, (21, 21), 0)

            mattes.append(matte)

        return mattes

    def deflicker(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Apply blind video deflickering for temporal consistency.

        Histogram matching between consecutive frames using a running average
        to smooth out brightness fluctuations.
        """
        if not frames or len(frames) < 2:
            return list(frames)

        result = [frames[0].copy()]

        # Compute running average of per-channel means
        running_mean = frames[0].astype(np.float64).mean(axis=(0, 1))
        alpha = 0.7  # smoothing factor — higher = more smoothing toward running avg

        for i in range(1, len(frames)):
            frame = frames[i].astype(np.float64)
            current_mean = frame.mean(axis=(0, 1))

            # Update running average
            running_mean = alpha * running_mean + (1 - alpha) * current_mean

            # Compute per-channel correction factors
            correction = np.where(
                current_mean > 0,
                running_mean / current_mean,
                1.0,
            )

            # Apply correction, clamp to [0, 255]
            corrected = frame * correction[np.newaxis, np.newaxis, :]
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            result.append(corrected)

        return result

    def encode_video(
        self,
        frames: list[np.ndarray],
        audio_path: Path,
        output_path: Path,
        fps: int = 25,
        resolution: tuple[int, int] = (1080, 1920),
        codec: str = "libx264",
        crf: int = 18,
    ) -> Path:
        """Encode frames + audio to final MP4 video via FFmpeg.

        Pipeline: frames -> resize if needed -> temp video -> mux with audio -> final MP4.
        """
        if not frames:
            raise ValueError("No frames to encode")

        # Resize if frames don't match target resolution
        target_w, target_h = resolution
        fh, fw = frames[0].shape[:2]
        if fw != target_w or fh != target_h:
            frames = resize_frames(frames, target_w, target_h)

        # Write frames to temp video (no audio)
        temp_dir = Path(tempfile.mkdtemp(prefix="petgen_"))
        temp_video = temp_dir / "temp_video.mp4"

        frames_to_video(frames, temp_video, fps=fps, codec=codec, crf=crf)

        # Mux audio with video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mux_audio_video(temp_video, audio_path, output_path)

        # Clean up temp file
        temp_video.unlink(missing_ok=True)
        temp_dir.rmdir()

        return output_path

    def blend_mouth_interior(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
        mouth_open_ref: np.ndarray,
        mouth_bbox: BoundingBox,
        blend_threshold: float = 0.3,
    ) -> list[np.ndarray]:
        """Blend pre-generated mouth-open reference into animated frames.

        When mouth opening exceeds blend_threshold (based on mask area ratio),
        alpha-blend the mouth_open_ref into the mouth region. This mitigates
        the "dark blob" artifact that appears when the animated mouth opens.

        Args:
            frames: List of RGB frames (H, W, 3).
            mouth_masks: Per-frame binary mouth masks.
            mouth_open_ref: RGB image of the character with mouth open.
            mouth_bbox: Bounding box of the mouth region in frame coordinates.
            blend_threshold: Minimum mouth openness ratio to trigger blending.

        Returns:
            List of frames with mouth interior blended where needed.
        """
        if not frames or not mouth_masks:
            return list(frames)

        # Pre-compute the mouth region crop from the reference
        bx1 = max(0, int(mouth_bbox.x1))
        by1 = max(0, int(mouth_bbox.y1))
        bx2 = min(mouth_open_ref.shape[1], int(mouth_bbox.x2))
        by2 = min(mouth_open_ref.shape[0], int(mouth_bbox.y2))
        bbox_area = max(1, (bx2 - bx1) * (by2 - by1))

        ref_crop = mouth_open_ref[by1:by2, bx1:bx2].copy()

        result = []
        for i, frame in enumerate(frames):
            if i >= len(mouth_masks):
                result.append(frame.copy())
                continue

            mask = mouth_masks[i]
            h, w = frame.shape[:2]

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Compute mouth openness as the ratio of mask area to bbox area
            if mask.dtype == np.uint8:
                mask_area = np.count_nonzero(mask[by1:by2, bx1:bx2])
            else:
                mask_area = np.sum(mask[by1:by2, bx1:bx2] > 0.5)

            openness = mask_area / bbox_area

            if openness < blend_threshold:
                result.append(frame.copy())
                continue

            # Blend strength ramps up from threshold to 1.0
            blend_alpha = min(1.0, (openness - blend_threshold) / (1.0 - blend_threshold))

            out = frame.copy()
            frame_crop = out[by1:by2, bx1:bx2]

            # Resize ref_crop to match frame_crop if needed
            rc = ref_crop
            if rc.shape[:2] != frame_crop.shape[:2]:
                rc = cv2.resize(rc, (frame_crop.shape[1], frame_crop.shape[0]))

            # Build feathered mask from mouth mask in the bbox region
            mask_region = mask[by1:by2, bx1:bx2].astype(np.float32)
            if mask_region.max() > 1.0:
                mask_region = mask_region / 255.0
            mask_region = cv2.GaussianBlur(mask_region, (11, 11), 0)
            mask_3ch = mask_region[:, :, np.newaxis] * blend_alpha

            # Alpha blend
            blended = (
                frame_crop.astype(np.float32) * (1 - mask_3ch)
                + rc.astype(np.float32) * mask_3ch
            ).astype(np.uint8)

            out[by1:by2, bx1:bx2] = blended
            result.append(out)

        return result

    def interpolate_frames(
        self,
        frames: list[np.ndarray],
        factor: int = 2,
    ) -> list[np.ndarray]:
        """Interpolate between frames using optical flow for smoother animation.

        Uses cv2.calcOpticalFlowFarneback to warp and blend consecutive frames,
        creating intermediate frames. Factor=2 doubles the frame count.

        Args:
            frames: List of RGB frames.
            factor: Interpolation factor (2 = double the frame count).

        Returns:
            Interpolated frame list with (len(frames) - 1) * factor + 1 frames.
        """
        if not frames or len(frames) < 2 or factor < 2:
            return list(frames)

        result = []

        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            result.append(f0.copy())

            gray0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray0, gray1, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            h, w = f0.shape[:2]
            # Generate intermediate frames
            for step in range(1, factor):
                t = step / factor
                # Forward warp from f0, backward warp from f1
                flow_t = flow * t
                flow_back = flow * (t - 1.0)

                # Build remap coordinates
                y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

                map_x_fwd = x_coords + flow_t[:, :, 0]
                map_y_fwd = y_coords + flow_t[:, :, 1]

                map_x_bwd = x_coords + flow_back[:, :, 0]
                map_y_bwd = y_coords + flow_back[:, :, 1]

                warped_fwd = cv2.remap(f0, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
                warped_bwd = cv2.remap(f1, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR)

                # Blend the two warps
                interp = (
                    warped_fwd.astype(np.float32) * (1 - t)
                    + warped_bwd.astype(np.float32) * t
                ).astype(np.uint8)

                result.append(interp)

        # Append the last frame
        result.append(frames[-1].copy())
        return result

    def composite_full(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
        audio_path: Path,
        output_path: Path,
        fps: int = 25,
        resolution: tuple[int, int] = (1080, 1920),
        deflicker_enabled: bool = True,
        inpainter: ProPainterInpainter | None = None,
        fur_matter: FurMatter | None = None,
        mouth_open_ref: np.ndarray | None = None,
        mouth_bbox: BoundingBox | None = None,
    ) -> Path:
        """Full compositing pipeline: repair -> matte -> deflicker -> encode.

        When optional upgraded modules are provided, they replace the baseline
        implementations:
        - inpainter: ProPainter replaces Gaussian-blur artifact repair
        - fur_matter: ViTMatte replaces Canny-edge alpha matte generation
        - mouth_open_ref + mouth_bbox: enables mouth interior blending
        """
        # Step 1: Artifact repair
        if inpainter is not None:
            result = inpainter.repair_mouth_artifacts(frames, mouth_masks)
        else:
            result = self.repair_artifacts(frames, mouth_masks)

        # Step 2: Alpha matting
        if fur_matter is not None:
            mattes = fur_matter.generate_mattes(result)
        else:
            mattes = self.generate_alpha_mattes(result)
        result = self.blend_fur_boundaries(result, mattes)

        # Step 3: Mouth interior blending
        if mouth_open_ref is not None and mouth_bbox is not None:
            result = self.blend_mouth_interior(
                result, mouth_masks, mouth_open_ref, mouth_bbox,
            )

        # Step 4: Deflicker
        if deflicker_enabled:
            result = self.deflicker(result)

        return self.encode_video(result, audio_path, output_path, fps=fps, resolution=resolution)
