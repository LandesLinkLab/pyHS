import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """Dual output logger - console and file"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.file = None
        
        if self.log_file:
            self.file = open(self.log_file, 'w', encoding='utf-8')
            # Write header
            self.file.write(f"{'='*60}\n")
            self.file.write(f"Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*60}\n\n")
            self.file.flush()
    
    def write(self, message):
        """Write to both console and file"""
        self.terminal.write(message)
        if self.file:
            self.file.write(message)
            self.file.flush()
    
    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        if self.file:
            self.file.flush()
    
    def close(self):
        """Close the log file"""
        if self.file:
            self.file.write(f"\n{'='*60}\n")
            self.file.write(f"Log ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*60}\n")
            self.file.close()
            self.file = None
    
    def __del__(self):
        """Ensure file is closed"""
        self.close()

# Global logger instance
_logger = None

def setup_logger(output_dir: str):
    """Setup global logger"""
    global _logger
    log_path = Path(output_dir) / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _logger = Logger(log_path)
    sys.stdout = _logger
    return _logger

def close_logger():
    """Close global logger and restore stdout"""
    global _logger
    if _logger:
        sys.stdout = _logger.terminal
        _logger.close()
        _logger = None
