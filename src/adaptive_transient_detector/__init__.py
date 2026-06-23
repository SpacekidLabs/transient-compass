from .config import DetectorConfig
from .detector import AdaptiveTransientDetector, FrameDecision

TransientDetector = AdaptiveTransientDetector

__all__ = ["AdaptiveTransientDetector", "DetectorConfig", "FrameDecision", "TransientDetector"]
