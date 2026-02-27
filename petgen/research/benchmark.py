"""Evaluation framework for comparing animation approaches."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PetGenBenchmark:
    """Evaluation framework for comparing animation approaches.

    Provides metrics for image quality, perceptual similarity,
    mouth-specific quality, and temporal consistency.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_fid(
        self, real_frames: list[np.ndarray], generated_frames: list[np.ndarray]
    ) -> float:
        """Frechet Inception Distance between real and generated frame sets.

        NOTE: Full FID requires pytorch-fid with InceptionV3 features.
        This implementation provides a simplified approximation using
        pixel-space statistics as a scaffold.

        Args:
            real_frames: List of real RGB frames (H, W, 3), uint8.
            generated_frames: List of generated RGB frames (H, W, 3), uint8.

        Returns:
            FID score (lower is better). Returns -1.0 if computation fails.
        """
        if not real_frames or not generated_frames:
            return -1.0

        # Compute mean and covariance of downsampled + channel-averaged features
        def _extract_features(frames: list[np.ndarray]) -> np.ndarray:
            features = []
            for f in frames:
                # Downsample to 16x16 and flatten to keep covariance tractable
                small = cv2.resize(f, (16, 16)).astype(np.float32).mean(axis=2).flatten() / 255.0
                features.append(small)
            return np.array(features)

        real_feat = _extract_features(real_frames)
        gen_feat = _extract_features(generated_frames)

        mu_r, mu_g = real_feat.mean(axis=0), gen_feat.mean(axis=0)
        sigma_r = np.cov(real_feat, rowvar=False) + np.eye(real_feat.shape[1]) * 1e-6
        sigma_g = np.cov(gen_feat, rowvar=False) + np.eye(gen_feat.shape[1]) * 1e-6

        # FID = ||mu_r - mu_g||^2 + Tr(sigma_r + sigma_g - 2*sqrtm(sigma_r @ sigma_g))
        diff = mu_r - mu_g
        mean_diff = float(np.dot(diff, diff))

        # Use eigenvalue-based approximation for the trace term
        # For identical distributions this correctly returns ~0
        try:
            product = sigma_r @ sigma_g
            eigvals = np.linalg.eigvalsh(product)
            eigvals = np.maximum(eigvals, 0)  # clamp negative eigenvalues
            sqrt_trace = float(np.sum(np.sqrt(eigvals)))
            trace_term = float(np.trace(sigma_r) + np.trace(sigma_g)) - 2 * sqrt_trace
        except np.linalg.LinAlgError:
            trace_term = 0.0

        fid = mean_diff + max(0.0, trace_term)
        return max(0.0, fid)

    def compute_lpips(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> float:
        """Learned Perceptual Image Patch Similarity.

        NOTE: Full LPIPS requires the lpips package with VGG/Alex features.
        This returns 0.0 as a stub. Install lpips for actual scores.

        Args:
            frame_a: RGB image (H, W, 3), uint8.
            frame_b: RGB image (H, W, 3), uint8.

        Returns:
            LPIPS distance (lower means more similar). Returns 0.0 in stub mode.
        """
        try:
            import lpips as lpips_pkg
            import torch

            loss_fn = lpips_pkg.LPIPS(net="alex")
            a = torch.from_numpy(frame_a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
            b = torch.from_numpy(frame_b).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
            return float(loss_fn(a, b).item())
        except ImportError:
            # Stub: return 0.0 — install lpips for real scores
            return 0.0

    def compute_mouth_ssim(
        self,
        real_frames: list[np.ndarray],
        gen_frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
    ) -> float:
        """SSIM specifically within the mouth region.

        Computes structural similarity only in the masked mouth area.

        Args:
            real_frames: Real RGB frames (H, W, 3), uint8.
            gen_frames: Generated RGB frames, same shape.
            mouth_masks: Binary masks (H, W), uint8, 255=mouth.

        Returns:
            Mean SSIM across all frame pairs within the mouth region.
        """
        if not real_frames or not gen_frames or not mouth_masks:
            return 0.0

        ssim_scores = []
        n = min(len(real_frames), len(gen_frames), len(mouth_masks))

        for i in range(n):
            real_gray = cv2.cvtColor(real_frames[i], cv2.COLOR_RGB2GRAY)
            gen_gray = cv2.cvtColor(gen_frames[i], cv2.COLOR_RGB2GRAY)
            mask = mouth_masks[i]

            # Only compute SSIM if there's a non-empty mouth region
            if mask.max() == 0:
                continue

            ssim_val = self._compute_ssim(real_gray, gen_gray, mask)
            ssim_scores.append(ssim_val)

        return float(np.mean(ssim_scores)) if ssim_scores else 0.0

    def _compute_ssim(
        self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray | None = None
    ) -> float:
        """Compute SSIM between two grayscale images, optionally within a mask.

        Uses the standard SSIM formula with window size 7.
        """
        try:
            from skimage.metrics import structural_similarity

            if mask is not None and mask.max() > 0:
                # Crop to mask bounding box for efficiency
                ys, xs = np.where(mask > 0)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                crop1 = img1[y1:y2, x1:x2]
                crop2 = img2[y1:y2, x1:x2]
                # Need minimum size for SSIM window
                if crop1.shape[0] < 7 or crop1.shape[1] < 7:
                    return float(1.0 - np.mean(np.abs(crop1.astype(float) - crop2.astype(float))) / 255.0)
                return float(structural_similarity(crop1, crop2))
            if img1.shape[0] < 7 or img1.shape[1] < 7:
                return float(1.0 - np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0)
            return float(structural_similarity(img1, img2))
        except ImportError:
            # Fallback: simple normalized MAE-based similarity
            diff = np.abs(img1.astype(float) - img2.astype(float))
            if mask is not None and mask.max() > 0:
                mask_bool = mask > 0
                diff = diff[mask_bool]
            return float(1.0 - diff.mean() / 255.0)

    def compute_temporal_consistency(self, frames: list[np.ndarray]) -> float:
        """Optical flow warping error between consecutive frames.

        Computes optical flow from frame[t] to frame[t+1], warps frame[t],
        and measures the MSE between warped and actual frame[t+1].
        Lower values indicate more temporally consistent animation.

        Args:
            frames: List of RGB frames (H, W, 3), uint8.

        Returns:
            Mean warping error across consecutive frame pairs.
            Lower = more temporally consistent.
        """
        if len(frames) < 2:
            return 0.0

        errors = []
        for i in range(len(frames) - 1):
            gray_a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Warp frame[t] using the flow field
            h, w = gray_a.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow[:, :, 0]).astype(np.float32)
            map_y = (map_y + flow[:, :, 1]).astype(np.float32)

            warped = cv2.remap(
                frames[i], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Compute MSE between warped and actual next frame
            mse = float(np.mean((warped.astype(float) - frames[i + 1].astype(float)) ** 2))
            errors.append(mse)

        return float(np.mean(errors))

    def compare_approaches(
        self,
        source_image: np.ndarray,
        audio_path: Path,
        warping_frames: list[np.ndarray],
        inpainting_frames: list[np.ndarray],
        ground_truth: list[np.ndarray] | None = None,
    ) -> dict:
        """Compare warping-based vs inpainting-based approaches.

        Args:
            source_image: Original source image.
            audio_path: Audio file used for animation.
            warping_frames: Frames from warping approach (e.g. LivePortrait).
            inpainting_frames: Frames from inpainting approach (e.g. MuseTalk).
            ground_truth: Optional ground truth frames for quality metrics.

        Returns:
            Dict with metrics for each approach.
        """
        results = {
            "warping": {
                "temporal_consistency": self.compute_temporal_consistency(warping_frames),
                "num_frames": len(warping_frames),
            },
            "inpainting": {
                "temporal_consistency": self.compute_temporal_consistency(inpainting_frames),
                "num_frames": len(inpainting_frames),
            },
        }

        if ground_truth:
            results["warping"]["fid"] = self.compute_fid(ground_truth, warping_frames)
            results["inpainting"]["fid"] = self.compute_fid(ground_truth, inpainting_frames)

            if warping_frames and ground_truth:
                results["warping"]["lpips_first"] = self.compute_lpips(
                    ground_truth[0], warping_frames[0]
                )
            if inpainting_frames and ground_truth:
                results["inpainting"]["lpips_first"] = self.compute_lpips(
                    ground_truth[0], inpainting_frames[0]
                )

        return results

    def generate_comparison_video(
        self,
        warping_frames: list[np.ndarray],
        inpainting_frames: list[np.ndarray],
        output_path: Path,
        fps: int = 25,
    ) -> Path:
        """Create side-by-side comparison video.

        Args:
            warping_frames: Frames from warping approach.
            inpainting_frames: Frames from inpainting approach.
            output_path: Where to save the comparison video.
            fps: Frames per second.

        Returns:
            Path to the created comparison video.
        """
        n = min(len(warping_frames), len(inpainting_frames))
        if n == 0:
            raise ValueError("No frames to compare")

        # Ensure same dimensions
        h = max(warping_frames[0].shape[0], inpainting_frames[0].shape[0])
        w = max(warping_frames[0].shape[1], inpainting_frames[0].shape[1])

        combined_frames = []
        for i in range(n):
            wf = cv2.resize(warping_frames[i], (w, h))
            inf = cv2.resize(inpainting_frames[i], (w, h))

            # Add labels
            labeled_w = wf.copy()
            labeled_i = inf.copy()
            cv2.putText(labeled_w, "Warping", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(labeled_i, "Inpainting", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Side by side
            combined = np.hstack([labeled_w, labeled_i])
            combined_frames.append(combined)

        # Write video using FFmpeg
        ch, cw = combined_frames[0].shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmp:
            tmp_path = tmp.name
            for frame in combined_frames:
                tmp.write(frame.tobytes())

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{cw}x{ch}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", tmp_path,
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        Path(tmp_path).unlink(missing_ok=True)

        logger.info("Comparison video saved to %s", output_path)
        return output_path
