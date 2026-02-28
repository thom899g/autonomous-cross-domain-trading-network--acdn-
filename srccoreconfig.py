"""
Configuration management for ACDN.
Centralizes all system configurations with environment-aware settings.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from dataclasses import dataclass, field
from enum import Enum

class ExchangeType(Enum):
    """Supported exchange types"""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    DERIVATIVES = "derivatives"

@dataclass
class ExchangeConfig:
    """Configuration for individual exchange connections"""
    name: str
    type: ExchangeType
    api_key_env: str
    api_secret_env: str
    rate_limit: int = 10  # requests per second
    timeout: int = 30  # seconds
    enabled: bool = True

@dataclass
class TradingConfig:
    """Trading strategy and risk parameters"""
    max_position_size_usd: float = 10000.0
    max_daily_loss_pct: float = 2.0
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0
    correlation_threshold: float = 0.7
    volatility_threshold: float = 0.15

@dataclass
class MLConfig:
    """Machine learning model configurations"""
    model_checkpoint_dir: str = "models/checkpoints"
    training_batch_size: int = 64
    prediction_batch_size: int = 32
    sequence_length: int = 100
    feature_count: int = 50

class Config:
    """Main configuration class with validation"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv("ACDN_ENV", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Base paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        
        # Trading configuration
        self.trading = TradingConfig()
        
        # ML configuration
        self.ml = MLConfig()
        
        # Exchange configurations
        self.exchanges = self._load_exchange_configs()
        
        # Firebase configuration
        self.firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
        
        # Validate
        self._validate()
        self._setup_directories()
    
    def _load_exchange_configs(self) -> Dict[str, ExchangeConfig]:
        """Load exchange configurations from environment"""
        exchanges = {}
        
        # Crypto exchanges
        exchanges["binance"] = ExchangeConfig(
            name="binance",
            type=ExchangeType.CRYPTO,
            api_key_env="BINANCE_API_KEY",
            api_secret_env="BINANCE_API_SECRET",
            rate_limit=20
        )
        
        exchanges["coinbase"] = ExchangeConfig(
            name="coinbase",
            type=ExchangeType.CRYPTO,
            api_key_env="COINBASE_API_KEY",
            api_secret_env="COINBASE_API_SECRET",
            rate_limit=15
        )
        
        # Stock exchanges (simulated via Alpaca)
        exchanges["alpaca"] = ExchangeConfig(
            name="alpaca",
            type=ExchangeType.STOCK,
            api_key_env="ALPACA_API_KEY",
            api_secret_env="ALPACA_API_SECRET",
            rate_limit=200
        )
        
        return exchanges
    
    def _validate(self):
        """Validate configuration"""
        if not self.firebase_credentials_path:
            logging.warning("Firebase credentials path not set. Firestore will be unavailable.")
        
        # Check required environment variables for enabled exchanges
        for exchange_name, config in self.exchanges.items():
            if config.enabled:
                if not os.getenv(config.api_key_env):
                    logging.warning(f"API key not set for {exchange_name}. Disabling.")
                    config.enabled = False
                if not os.getenv(config.api_secret_env):
                    logging.warning(f"API secret not set for {exchange_name}. Disabling.")
                    config.enabled = False
    
    def _setup_directories(self):
        """Create required directories"""
        directories = [self.data_dir, self.logs_dir, self.models_dir]
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
    
    def get_enabled_exchanges(self) -> List[ExchangeConfig]:
        """Get list of enabled exchange configurations"""
        return [config for config in self.exchanges.values() if config.enabled]

# Global configuration instance
config = Config()