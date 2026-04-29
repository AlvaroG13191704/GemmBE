# MicroTRIBE-Gemma: Simplified TriBE v2 with Gemma 4 E2B
# Brain encoding architecture using early-fusion multimodal representations.

from src.config import ModelConfig
from src.model import MicroTribeGemma
from src.tail_model import TailModel
from src.subject_block import Bottleneck, SubjectBlock, MultiSubjectBlock
from src.temporal_alignment import TemporalPooling, HRFAligner
from src.dataset import PreExtractedDataset, MultiSubjectDataset
from src.extract_features import OfflineExtractor, generate_real_extraction

__all__ = [
    # Config
    "ModelConfig",
    # Models
    "MicroTribeGemma",       # Modelo completo (con Gemma 4)
    "TailModel",             # Modelo ligero (sin Gemma 4, para offline)
    # Components
    "Bottleneck",
    "SubjectBlock",
    "MultiSubjectBlock",
    "TemporalPooling",
    "HRFAligner",
    # Data
    "PreExtractedDataset",
    "MultiSubjectDataset",
    "OfflineExtractor",
    "generate_real_extraction",
]
