"""PetGen pipeline modules."""

from petgen.modules.animation import LipSyncAnimator
from petgen.modules.character import CharacterCreator
from petgen.modules.compositing import Compositor
from petgen.modules.face_detect import PetFaceDetector
from petgen.modules.motion import SecondaryMotion
from petgen.modules.voice import VoiceGenerator

__all__ = [
    "CharacterCreator",
    "VoiceGenerator",
    "PetFaceDetector",
    "LipSyncAnimator",
    "SecondaryMotion",
    "Compositor",
]
