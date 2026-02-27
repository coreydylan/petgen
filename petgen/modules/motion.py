"""Secondary motion system — breathing, ear twitches, blinks, head bob, tail wag."""

from __future__ import annotations

import cv2
import numpy as np

from petgen.models import AnimationConfig, BreedGroup, Landmark


class SecondaryMotion:
    """Applies procedural secondary motion to animated frames for realism."""

    def apply_breathing(
        self,
        frames: list[np.ndarray],
        bpm: float = 15.0,
        amplitude: float = 0.02,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply sinusoidal breathing motion to chest/torso region.

        Dogs: 12-18 BPM, Cats: 20-30 BPM. Amplitude 1-3% scale oscillation.
        Warps the lower 60% of the frame with a vertical scale, leaving the face untouched.
        """
        if not frames:
            return frames

        result = []
        h, w = frames[0].shape[:2]
        # The boundary between the untouched upper region and the breathing zone
        split_y = int(h * 0.4)

        for i, frame in enumerate(frames):
            t = i / fps
            scale = 1.0 + amplitude * np.sin(2 * np.pi * (bpm / 60.0) * t)

            # Keep the upper portion (face) untouched
            upper = frame[:split_y]

            # Warp the lower portion (torso/chest) with vertical scaling
            lower = frame[split_y:]
            lh, lw = lower.shape[:2]

            # Scale from the top edge of the lower region (anchored at the split line)
            # Translate so top stays pinned, scale vertically
            cx = lw / 2.0
            M = np.array([
                [1.0, 0.0, 0.0],
                [0.0, scale, -(scale - 1.0) * 0.0],  # anchor at top of lower region
            ], dtype=np.float64)
            # We want the top row of lower to stay fixed, so the y-translation is 0
            warped_lower = cv2.warpAffine(
                lower, M, (lw, lh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            out = np.vstack([upper, warped_lower])
            result.append(out)

        return result

    def apply_ear_twitches(
        self,
        frames: list[np.ndarray],
        audio_energy: np.ndarray,
        landmarks: list[list[Landmark]],
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply ear twitches synchronized to speech emphasis.

        When audio energy spikes, apply small oscillation to the ear region.
        Random micro-twitches between words when energy is low.
        Uses local affine warps on the ear ROI (indices 0-3 for ear tips).
        """
        if not frames or len(landmarks) == 0:
            return frames

        rng = np.random.RandomState(42)
        result = []

        # Normalize energy for threshold detection
        energy = np.asarray(audio_energy, dtype=np.float64)
        if len(energy) == 0:
            return list(frames)
        energy_mean = energy.mean() if energy.mean() > 0 else 1.0
        energy_norm = energy / energy_mean

        for i, frame in enumerate(frames):
            if i >= len(landmarks) or len(landmarks[i]) < 4:
                result.append(frame.copy())
                continue

            # Get ear-tip keypoints (indices 0-3)
            ear_pts = landmarks[i][:4]
            energy_idx = min(i, len(energy_norm) - 1)
            e = energy_norm[energy_idx]

            # Determine displacement: larger on energy spikes, micro-twitches when low
            if e > 1.5:
                # Emphasis spike — stronger twitch
                displacement = rng.uniform(2.0, 5.0)
            elif e < 0.3:
                # Low energy — occasional micro-twitch
                displacement = rng.uniform(0.0, 2.0) if rng.random() < 0.15 else 0.0
            else:
                displacement = 0.0

            if displacement < 0.5:
                result.append(frame.copy())
                continue

            h, w = frame.shape[:2]
            out = frame.copy()

            # Build source and destination points for the ear region affine warp
            src_pts = np.array(
                [[lm.x, lm.y] for lm in ear_pts[:3]], dtype=np.float32,
            )
            # Apply a small vertical/horizontal oscillation
            t = i / fps
            dx = displacement * np.sin(2 * np.pi * 3.0 * t)
            dy = displacement * np.cos(2 * np.pi * 2.5 * t) * 0.5
            dst_pts = src_pts.copy()
            dst_pts[:, 0] += dx
            dst_pts[:, 1] += dy

            # Compute the bounding box of ear points for a local warp
            xs = [lm.x for lm in ear_pts]
            ys = [lm.y for lm in ear_pts]
            pad = 30
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)

            if x2 - x1 < 10 or y2 - y1 < 10:
                result.append(out)
                continue

            roi = out[y1:y2, x1:x2].copy()

            # Shift source/dest into ROI coordinates
            offset = np.array([x1, y1], dtype=np.float32)
            src_local = src_pts - offset
            dst_local = dst_pts - offset

            M = cv2.getAffineTransform(src_local, dst_local)
            warped_roi = cv2.warpAffine(
                roi, M, (roi.shape[1], roi.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Alpha blend the warped region back to avoid hard edges
            mask = np.ones(roi.shape[:2], dtype=np.float32)
            kernel_size = min(7, max(3, pad // 3))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            mask = mask[:, :, np.newaxis]
            blended = (warped_roi * mask + roi * (1 - mask)).astype(np.uint8)
            out[y1:y2, x1:x2] = blended

            result.append(out)

        return result

    def apply_head_bob(
        self,
        frames: list[np.ndarray],
        audio_pitch: np.ndarray,
        amplitude: float = 0.01,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply gentle head nodding on stressed syllables.

        Uses pitch contour to detect peaks (stressed syllables) and applies
        a vertical translation to the upper portion of the frame.
        """
        if not frames:
            return frames

        pitch = np.asarray(audio_pitch, dtype=np.float64)
        if len(pitch) == 0:
            return list(frames)

        # Normalize pitch to [0, 1] range for displacement scaling
        p_min = pitch.min()
        p_max = pitch.max()
        if p_max - p_min > 0:
            pitch_norm = (pitch - p_min) / (p_max - p_min)
        else:
            pitch_norm = np.zeros_like(pitch)

        result = []
        h, w = frames[0].shape[:2]
        max_dy = amplitude * h  # max displacement in pixels

        for i, frame in enumerate(frames):
            pitch_idx = min(i, len(pitch_norm) - 1)
            p = pitch_norm[pitch_idx]

            # Vertical displacement proportional to pitch
            dy = max_dy * p

            if abs(dy) < 0.3:
                result.append(frame.copy())
                continue

            # Translate the upper 50% of the frame
            split_y = h // 2
            upper = frame[:split_y]
            uh, uw = upper.shape[:2]

            M = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, dy],
            ], dtype=np.float64)
            warped_upper = cv2.warpAffine(
                upper, M, (uw, uh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            out = frame.copy()
            out[:split_y] = warped_upper
            result.append(out)

        return result

    def apply_blinks(
        self,
        frames: list[np.ndarray],
        landmarks: list[list[Landmark]],
        interval_range: tuple[float, float] = (3.0, 6.0),
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply random blinks every 3-6 seconds.

        Quick close (2-3 frames) + slower open (4-5 frames).
        10% chance of double blink for personality.
        Scales the eye region vertically using landmarks to identify eye positions.
        """
        if not frames or len(landmarks) == 0:
            return frames

        rng = np.random.RandomState(123)
        num_frames = len(frames)
        total_duration = num_frames / fps

        # Schedule blink times
        blink_times: list[float] = []
        t = rng.uniform(interval_range[0], interval_range[1])
        while t < total_duration:
            blink_times.append(t)
            # 10% chance of double blink
            if rng.random() < 0.1:
                blink_times.append(t + 0.15)  # quick follow-up
            t += rng.uniform(interval_range[0], interval_range[1])

        # Convert blink times to frame indices
        blink_frames: list[int] = [int(bt * fps) for bt in blink_times]

        # Build per-frame blink scale: 1.0 = open, 0.0 = fully closed
        blink_scale = np.ones(num_frames, dtype=np.float64)
        close_dur = 3  # frames to close
        open_dur = 5   # frames to open

        for bf in blink_frames:
            if bf >= num_frames:
                continue
            # Closing phase
            for j in range(close_dur):
                idx = bf + j
                if idx < num_frames:
                    # Linearly close: 1 -> 0
                    s = 1.0 - (j + 1) / close_dur
                    blink_scale[idx] = min(blink_scale[idx], s)
            # Opening phase
            for j in range(open_dur):
                idx = bf + close_dur + j
                if idx < num_frames:
                    s = (j + 1) / open_dur
                    blink_scale[idx] = min(blink_scale[idx], s)

        result = []
        for i, frame in enumerate(frames):
            s = blink_scale[i]
            if s >= 0.99:
                result.append(frame.copy())
                continue

            h, w = frame.shape[:2]

            # Determine eye region from landmarks (indices 4-7 if available, else estimate)
            if i < len(landmarks) and len(landmarks[i]) >= 8:
                eye_lms = landmarks[i][4:8]
                eys = [lm.y for lm in eye_lms]
                exs = [lm.x for lm in eye_lms]
                eye_cy = int(np.mean(eys))
                eye_cx = int(np.mean(exs))
            else:
                # Estimate: eyes are roughly at 30% from top, centered
                eye_cy = int(h * 0.30)
                eye_cx = w // 2

            # Define a region around the eyes to scale vertically
            eye_h = int(h * 0.12)  # region height
            eye_w = int(w * 0.5)   # region width
            y1 = max(0, eye_cy - eye_h)
            y2 = min(h, eye_cy + eye_h)
            x1 = max(0, eye_cx - eye_w // 2)
            x2 = min(w, eye_cx + eye_w // 2)

            roi = frame[y1:y2, x1:x2].copy()
            rh, rw = roi.shape[:2]

            if rh < 4 or rw < 4:
                result.append(frame.copy())
                continue

            # Scale the eye ROI vertically toward the center to simulate closing
            center_y = rh / 2.0
            M = np.array([
                [1.0, 0.0, 0.0],
                [0.0, s, center_y * (1.0 - s)],
            ], dtype=np.float64)
            warped_roi = cv2.warpAffine(
                roi, M, (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            out = frame.copy()
            out[y1:y2, x1:x2] = warped_roi
            result.append(out)

        return result

    def apply_tail_wag(
        self,
        frames: list[np.ndarray],
        excitement: float = 0.5,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply procedural pendulum tail motion if tail is visible.

        Sinusoidal warp on the lower portion of the frame. Frequency increases
        with excitement level.
        """
        if not frames:
            return frames

        result = []
        h, w = frames[0].shape[:2]

        # Tail region: bottom 25% of the frame
        split_y = int(h * 0.75)

        # Frequency scales with excitement: 0.5 Hz at calm to 3.0 Hz at max excitement
        freq = 0.5 + 2.5 * np.clip(excitement, 0.0, 1.0)
        max_angle_deg = 5.0 + 10.0 * np.clip(excitement, 0.0, 1.0)

        for i, frame in enumerate(frames):
            t = i / fps
            angle = max_angle_deg * np.sin(2 * np.pi * freq * t)

            lower = frame[split_y:]
            lh, lw = lower.shape[:2]

            if lh < 4 or lw < 4:
                result.append(frame.copy())
                continue

            # Rotate the lower region around its top-center point
            center = (lw / 2.0, 0.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            warped_lower = cv2.warpAffine(
                lower, M, (lw, lh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            out = frame.copy()
            out[split_y:] = warped_lower
            result.append(out)

        return result

    def apply_all(
        self,
        frames: list[np.ndarray],
        audio_features: dict[str, np.ndarray],
        landmarks: list[list[Landmark]],
        config: AnimationConfig,
        breed_group: BreedGroup = BreedGroup.MEDIUM_DOG,
    ) -> list[np.ndarray]:
        """Apply all enabled secondary motion effects."""
        fps = float(config.fps)

        # Determine breathing rate from breed
        if config.breathing_bpm:
            bpm = config.breathing_bpm
        elif breed_group in (BreedGroup.CAT, BreedGroup.KITTEN):
            bpm = 25.0
        else:
            bpm = 15.0

        result = frames
        result = self.apply_breathing(result, bpm=bpm, fps=fps)

        if config.enable_blinks:
            result = self.apply_blinks(
                result, landmarks, interval_range=config.blink_interval, fps=fps
            )

        if config.enable_ear_twitches and "energy" in audio_features:
            result = self.apply_ear_twitches(result, audio_features["energy"], landmarks, fps=fps)

        if config.enable_head_bob and "pitch" in audio_features:
            result = self.apply_head_bob(result, audio_features["pitch"], fps=fps)

        if config.enable_tail_wag:
            excitement = audio_features.get("mean_energy", np.array([0.5])).mean()
            result = self.apply_tail_wag(result, excitement=float(excitement), fps=fps)

        return result
