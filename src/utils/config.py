"""
Configuration management using YAML and environment variables
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        return self._override_with_env(config)
    
    def _override_with_env(self, config: Dict) -> Dict:
        """Override config values with environment variables"""
        for key, value in config.items():
            env_key = key.upper()
            if env_key in os.environ:
                config[key] = os.environ[env_key]
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
