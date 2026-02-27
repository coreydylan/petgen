"""Lip sync animation using JoyVASA (audio->motion) + LivePortrait (motion->frames)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import AnimationConfig, MotionSequence

logger = logging.getLogger(__name__)

# Default number of implicit keypoints used by LivePortrait animal mode
_NUM_KEYPOINTS = 21
# Dimension of expression coefficients vector
_EXPRESSION_DIM = 63
# Indices within the expression vector that control mouth opening.
# These correspond to jaw-open and lip-separation blendshapes in the
# LivePortrait animal rig.
_MOUTH_EXPRESSION_INDICES = list(range(33, 48))


class LipSyncAnimator:
    """Audio-driven face animation for pets using JoyVASA + LivePortrait."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._joyvasa = None
        self._liveportrait = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_joyvasa(self):
        """Lazy-load the JoyVASA audio-to-motion pipeline.

        JoyVASA consists of:
        1. wav2vec2 encoder — extracts audio features
        2. Diffusion transformer — generates identity-independent motion sequences

        Weights are expected under ``config.weights_dir / 'joyvasa/'``.
        """
        if self._joyvasa is not None:
            return self._joyvasa

        joyvasa_dir = self.config.weights_dir / "joyvasa"
        if not joyvasa_dir.exists():
            raise FileNotFoundError(
                f"JoyVASA weights not found at {joyvasa_dir}. "
                "Download from https://github.com/jdh-algo/JoyVASA"
            )

        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model

        device = self.config.device
        dtype = torch.float16 if self.config.half_precision and device != "cpu" else torch.float32

        # Load wav2vec2 for audio feature extraction
        wav2vec_path = joyvasa_dir / "wav2vec2"
        if wav2vec_path.exists():
            processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
            wav2vec = Wav2Vec2Model.from_pretrained(str(wav2vec_path)).to(device, dtype)
        else:
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(
                device, dtype
            )

        wav2vec.eval()

        # Load the diffusion transformer for motion generation
        diff_path = joyvasa_dir / "motion_diffusion.pt"
        diff_model = torch.jit.load(str(diff_path), map_location=device)
        diff_model.eval()
        if self.config.half_precision and device != "cpu":
            diff_model = diff_model.half()

        self._joyvasa = {
            "processor": processor,
            "wav2vec": wav2vec,
            "diffusion": diff_model,
            "device": device,
            "dtype": dtype,
        }
        logger.info("JoyVASA pipeline loaded on device=%s dtype=%s", device, dtype)
        return self._joyvasa

    def _load_liveportrait(self):
        """Lazy-load the LivePortrait animal-mode warping pipeline.

        LivePortrait uses:
        1. Appearance encoder — extracts source identity features
        2. Motion encoder — extracts canonical keypoints
        3. Warping module — applies motion to source appearance
        4. SPADE generator — decodes warped features into pixels

        Weights are expected under ``config.weights_dir / 'liveportrait_animal/'``.
        """
        if self._liveportrait is not None:
            return self._liveportrait

        lp_dir = self.config.weights_dir / "liveportrait_animal"
        if not lp_dir.exists():
            raise FileNotFoundError(
                f"LivePortrait animal weights not found at {lp_dir}. "
                "Download from https://github.com/KwaiVGI/LivePortrait"
            )

        import torch

        device = self.config.device
        dtype = torch.float16 if self.config.half_precision and device != "cpu" else torch.float32

        # Load sub-networks
        appearance_encoder = torch.jit.load(
            str(lp_dir / "appearance_encoder.pt"), map_location=device
        )
        motion_encoder = torch.jit.load(
            str(lp_dir / "motion_encoder.pt"), map_location=device
        )
        warping_module = torch.jit.load(
            str(lp_dir / "warping_module.pt"), map_location=device
        )
        spade_generator = torch.jit.load(
            str(lp_dir / "spade_generator.pt"), map_location=device
        )

        for m in (appearance_encoder, motion_encoder, warping_module, spade_generator):
            m.eval()
            if self.config.half_precision and device != "cpu":
                m.half()

        self._liveportrait = {
            "appearance_encoder": appearance_encoder,
            "motion_encoder": motion_encoder,
            "warping_module": warping_module,
            "spade_generator": spade_generator,
            "device": device,
            "dtype": dtype,
        }
        logger.info("LivePortrait animal pipeline loaded on device=%s", device)
        return self._liveportrait

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_motion(
        self,
        audio_path: Path,
        source_image: Path,
    ) -> MotionSequence:
        """Generate motion sequences from audio using JoyVASA diffusion transformer.

        Uses wav2vec2 for audio features -> diffusion transformer -> identity-independent
        motion sequences compatible with LivePortrait.

        Args:
            audio_path: Path to the audio file (WAV preferred, any ffmpeg format OK).
            source_image: Path to the source portrait (used for duration reference only;
                          the motion is identity-independent).

        Returns:
            MotionSequence with keypoints (T, N, 2) and expressions (T, D).
        """
        import torch
        import torchaudio

        joyvasa = self._load_joyvasa()
        device = joyvasa["device"]
        dtype = joyvasa["dtype"]

        # Load and resample audio to 16 kHz (wav2vec2 requirement)
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract wav2vec2 features
        processor = joyvasa["processor"]
        wav2vec = joyvasa["wav2vec"]
        inputs = processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device, dtype)

        with torch.no_grad():
            audio_features = wav2vec(input_values).last_hidden_state  # (1, T_audio, D)

        # Run diffusion transformer to generate motion
        diffusion = joyvasa["diffusion"]
        with torch.no_grad():
            motion_output = diffusion(audio_features)  # (1, T, N*2 + D_expr)

        motion_np = motion_output.float().cpu().numpy()[0]  # (T, N*2 + D_expr)

        # Split into keypoints and expressions
        kp_dim = _NUM_KEYPOINTS * 2
        keypoints = motion_np[:, :kp_dim].reshape(-1, _NUM_KEYPOINTS, 2)
        expressions = motion_np[:, kp_dim : kp_dim + _EXPRESSION_DIM]

        return MotionSequence(
            keypoints=keypoints,
            expressions=expressions,
            fps=25.0,
        )

    def animate(
        self,
        source_image: Path,
        motion: MotionSequence,
        config: AnimationConfig | None = None,
    ) -> list[np.ndarray]:
        """Apply motion sequences to source image using LivePortrait animal mode.

        Uses implicit keypoint-based warping with SPADE generator.
        CRITICAL: Uses no_flag_stitching=True for animals (stitching module
        was not trained on animal data).

        Args:
            source_image: Path to the source pet portrait.
            motion: Motion sequence from generate_motion().
            config: Optional animation config for resolution etc.

        Returns:
            List of RGB frames as numpy arrays (H, W, 3), uint8.
        """
        import cv2
        import torch

        lp = self._load_liveportrait()
        device = lp["device"]
        dtype = lp["dtype"]

        # Load and preprocess source image
        img = cv2.imread(str(source_image))
        if img is None:
            raise FileNotFoundError(f"Cannot read source image: {source_image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to LivePortrait input resolution (256x256 internal)
        orig_h, orig_w = img.shape[:2]
        lp_size = 256
        img_resized = cv2.resize(img, (lp_size, lp_size))
        img_tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device, dtype)
            / 255.0
        )

        # Encode source appearance (done once)
        appearance_encoder = lp["appearance_encoder"]
        motion_encoder = lp["motion_encoder"]
        warping_module = lp["warping_module"]
        spade_generator = lp["spade_generator"]

        with torch.no_grad():
            source_features = appearance_encoder(img_tensor)
            source_kp = motion_encoder(img_tensor)  # canonical keypoints

        # Determine output resolution
        if config:
            out_w, out_h = config.resolution
        else:
            out_w, out_h = orig_w, orig_h

        frames: list[np.ndarray] = []

        # Animate frame by frame
        for t in range(motion.num_frames):
            kp_drive = torch.from_numpy(motion.keypoints[t : t + 1]).to(device, dtype)
            expr_drive = torch.from_numpy(motion.expressions[t : t + 1]).to(device, dtype)

            with torch.no_grad():
                # Warp source features using driving motion
                # no_flag_stitching=True is critical for animals
                warped = warping_module(source_features, source_kp, kp_drive, expr_drive)
                out_tensor = spade_generator(warped)  # (1, 3, 256, 256)

            # Convert to numpy RGB frame
            frame = (
                out_tensor[0]
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            # Resize to target resolution if needed
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h))

            frames.append(frame)

            # Periodically clear GPU cache to manage memory
            if t % 50 == 49 and device != "cpu":
                torch.cuda.empty_cache()

        return frames

    def animate_from_audio(
        self,
        source_image: Path,
        audio_path: Path,
        config: AnimationConfig | None = None,
    ) -> list[np.ndarray]:
        """End-to-end: audio + source image -> animated frames."""
        motion = self.generate_motion(audio_path, source_image)
        if config and config.mouth_amplitude_limit < 1.0:
            motion = self._limit_mouth_amplitude(motion, config.mouth_amplitude_limit)
        return self.animate(source_image, motion, config)

    def _limit_mouth_amplitude(
        self,
        motion: MotionSequence,
        limit: float,
    ) -> MotionSequence:
        """Clamp mouth opening amplitude to reduce dark-blob artifacts.

        Scales the mouth-related expression coefficients and keypoint deltas
        by the *limit* factor (0-1).  A limit of 0.8 means the mouth opens
        to at most 80% of the original predicted range.

        The original MotionSequence is not mutated; a new copy is returned.
        """
        if limit < 0.0 or limit > 1.0:
            raise ValueError(f"mouth amplitude limit must be in [0, 1], got {limit}")

        # Copy arrays so we don't mutate the input
        new_kp = motion.keypoints.copy()
        new_expr = motion.expressions.copy()

        # Scale mouth expression coefficients
        for idx in _MOUTH_EXPRESSION_INDICES:
            if idx < new_expr.shape[1]:
                # Compute per-channel mean, then scale deviation from mean
                mean_val = new_expr[:, idx].mean()
                new_expr[:, idx] = mean_val + (new_expr[:, idx] - mean_val) * limit

        # Scale mouth-related keypoint deltas.
        # Mouth keypoints in LivePortrait animal mode are approximately the
        # last 5 of the 21 implicit keypoints.
        mouth_kp_start = _NUM_KEYPOINTS - 5  # indices 16-20
        if new_kp.shape[1] > mouth_kp_start:
            # Compute temporal mean per mouth keypoint and scale deviation
            mean_kp = new_kp[:, mouth_kp_start:, :].mean(axis=0, keepdims=True)
            new_kp[:, mouth_kp_start:, :] = (
                mean_kp + (new_kp[:, mouth_kp_start:, :] - mean_kp) * limit
            )

        return MotionSequence(
            keypoints=new_kp,
            expressions=new_expr,
            fps=motion.fps,
        )
