# -*- coding: utf-8 -*-
"""Core package of the best fall-detection combination (A2).

Only three capabilities are exposed:
  - BestFallModel        : RF(400) + XGB(400) probability-averaging classifier (save / load)
  - detect_offline       : offline detection of a whole recording (sample data / history replay)
  - StreamingDetector    : real-time detection on a continuous sensor stream

Hyper-parameters are in `config.py`; the package contains no experimental candidate methods, only the best path found in the experiments.
"""
from . import config          # noqa: F401
from .model import BestFallModel, feature_importance          # noqa: F401
from .detector import StreamingDetector, detect_offline       # noqa: F401

__all__ = ["config", "BestFallModel", "feature_importance",
           "StreamingDetector", "detect_offline"]
