from dataclasses import dataclass, field
import time
import uuid

@dataclass
class DetectionEvent:
    """A single observation made by a worker camera."""
    camera_id: str
    timestamp: float
    bbox: list              # [x1, y1, x2, y2]
    appearance_vector: dict # The output from feature_extractor.py
    
@dataclass
class Trail:
    """A story built from multiple Detection Events."""
    trail_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    events: list[DetectionEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: str = "active"  # 'active', 'closed', or 'incident'
    
    def add_event(self, event: DetectionEvent):
        self.events.append(event)
        self.last_updated = event.timestamp
        
    def get_path_summary(self):
        """Returns a string like: 'gate -> corridor'"""
        return " -> ".join([e.camera_id for e in self.events])