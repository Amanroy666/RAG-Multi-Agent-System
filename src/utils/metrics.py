"""
Metrics collection and monitoring
"""
from typing import Dict, Any
from datetime import datetime
import time
from functools import wraps
from .logger import setup_logger

logger = setup_logger(__name__)

class MetricsCollector:
    """Collect and track application metrics"""
    
    def __init__(self):
        self.metrics = {
            'counters': {},
            'gauges': {},
            'histograms': {}
        }
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict = None):
        """Increment a counter metric"""
        key = self._make_key(name, tags)
        self.metrics['counters'][key] = self.metrics['counters'].get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, tags: Dict = None):
        """Set a gauge metric"""
        key = self._make_key(name, tags)
        self.metrics['gauges'][key] = value
    
    def record_histogram(self, name: str, value: float, tags: Dict = None):
        """Record a histogram value"""
        key = self._make_key(name, tags)
        if key not in self.metrics['histograms']:
            self.metrics['histograms'][key] = []
        self.metrics['histograms'][key].append(value)
    
    def _make_key(self, name: str, tags: Dict = None) -> str:
        """Create metric key from name and tags"""
        if not tags:
            return name
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.metrics

def timing_metric(metric_name: str):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.info(f"{metric_name}: {duration:.3f}s")
        return wrapper
    return decorator
