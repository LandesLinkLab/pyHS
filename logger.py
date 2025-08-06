import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """
    Dual output logger that writes to both console and file simultaneously
    
    This class captures all print statements and console output, redirecting
    them to both the terminal (for real-time monitoring) and a log file 
    (for permanent record keeping). This is essential for long-running
    analysis pipelines where you need both immediate feedback and a 
    complete record of the analysis process.
    
    The logger automatically adds timestamps and formatting to create
    professional log files that can be used for troubleshooting,
    result verification, and analysis documentation.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize the dual logger system
        
        Parameters:
        -----------
        log_file : Optional[Path]
            Path to the log file. If None, only console output is used.
            If provided, all output will be written to both console and file.
        """
        # Store original stdout for console output
        self.terminal = sys.stdout
        self.log_file = log_file
        self.file = None
        
        # Open log file if path provided
        if self.log_file:
            self.file = open(self.log_file, 'w', encoding='utf-8')
            
            # Write professional header to log file
            self.file.write(f"{'='*60}\n")
            self.file.write(f"Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*60}\n\n")
            self.file.flush()  # Ensure header is written immediately
    
    def write(self, message):
        """
        Write message to both console and log file
        
        This method is called automatically by Python when print() is used,
        since we redirect sys.stdout to this logger instance.
        
        Parameters:
        -----------
        message : str
            Text message to be written to both outputs
        """
        # Always write to console for real-time feedback
        self.terminal.write(message)
        
        # Also write to file if available
        if self.file:
            self.file.write(message)
            self.file.flush()  # Ensure immediate writing (important for long analyses)
    
    def flush(self):
        """
        Flush both output streams to ensure all data is written
        
        This is called by Python's output system and ensures that
        buffered data is written immediately to both console and file.
        """
        self.terminal.flush()
        if self.file:
            self.file.flush()
    
    def close(self):
        """
        Properly close the log file with footer information
        
        This method adds a professional footer to the log file and
        closes the file handle. It's safe to call multiple times.
        """
        if self.file:
            # Add footer with completion timestamp
            self.file.write(f"\n{'='*60}\n")
            self.file.write(f"Log ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*60}\n")
            self.file.close()
            self.file = None
    
    def __del__(self):
        """
        Destructor to ensure file is closed even if explicit close() isn't called
        
        This provides a safety net to prevent leaving log files open
        if the logger object goes out of scope unexpectedly.
        """
        self.close()

# Global logger instance for the application
_logger = None

def setup_logger(output_dir: str):
    """
    Set up global dual logging system for the entire application
    
    This function:
    1. Creates the output directory if it doesn't exist
    2. Creates a Logger instance that writes to both console and file
    3. Redirects sys.stdout to the logger so all print statements are captured
    4. Returns the logger instance for manual control if needed
    
    Parameters:
    -----------
    output_dir : str
        Directory where the log file will be created
    
    Returns:
    --------
    Logger
        The logger instance that was created and activated
        
    Notes:
    ------
    After calling this function, all print() statements and stdout output
    will automatically be written to both console and the log file.
    Call close_logger() to restore normal output and close the log file.
    """
    global _logger
    
    # Create log file path
    log_path = Path(output_dir) / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    
    # Create logger instance
    _logger = Logger(log_path)
    
    # Redirect stdout to the logger so all print statements are captured
    sys.stdout = _logger
    
    return _logger

def close_logger():
    """
    Close the global logging system and restore normal console output
    
    This function:
    1. Restores sys.stdout to its original state (console only)
    2. Closes the log file properly with footer information
    3. Cleans up the global logger instance
    
    This should be called at the end of analysis or in finally blocks
    to ensure proper cleanup even if the analysis fails partway through.
    
    It's safe to call this function multiple times - subsequent calls
    will have no effect if the logger is already closed.
    """
    global _logger
    
    if _logger:
        # Restore original stdout
        sys.stdout = _logger.terminal
        
        # Close log file properly
        _logger.close()
        
        # Clear global reference
        _logger = None