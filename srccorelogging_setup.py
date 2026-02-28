"""
Robust logging configuration for ACDN.
Provides structured logging with multiple handlers and Firestore integration.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime
import json
from .config import config

class FirestoreLogHandler(logging.Handler):
    """Custom log handler for Firestore integration"""
    
    def __init__(self, firestore_client=None):
        super().__init__()
        self.firestore_client = firestore_client
        self.batch_size = 100
        self._buffer = []
    
    def emit(self, record):
        """