import sys
from pathlib import Path
from datetime import datetime
import logging
import atexit
from typing import Optional

class BufferedTeeWriter:
    def __init__(self, filepath: Path, original_stream):
        self.file = open(filepath, 'a', buffering=8192)  # Append mode with buffering
        self.original_stream = original_stream
        atexit.register(self.cleanup)
        
    def write(self, data: str) -> None:
        try:
            self.file.write(data)
            self.original_stream.write(data)
        except (IOError, OSError) as e:
            logging.error(f"Failed to write to log: {e}")
            
    def flush(self) -> None:
        if self.file and not self.file.closed:
            try:
                self.file.flush()
                self.original_stream.flush()
            except (IOError, OSError) as e:
                logging.error(f"Failed to flush log: {e}")
                
    def cleanup(self) -> None:
        if self.file and not self.file.closed:
            try:
                self.file.flush()
                self.file.close()
            except (IOError, OSError):
                pass

def setup_logging(log_dir: str = "logs") -> Optional[Path]:
    """
    Set up logging with a single append-only log file.
    
    Args:
        log_dir: Directory to store the log file
    
    Returns:
        Path to the log file or None if setup fails
    """
    try:
        # Create logs directory if needed
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Use a single log file
        stderr_log = log_path / "stderr.log"
        
        # Add a session start marker to the log
        with open(stderr_log, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"New logging session started at {datetime.now()}\n")
            f.write(f"{'='*80}\n\n")
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set up buffered stderr redirection
        sys.stderr = BufferedTeeWriter(stderr_log, sys.__stderr__)
        
        return stderr_log
        
    except Exception as e:
        logging.error(f"Failed to setup logging: {e}")
        return None

# Optional: Function to rotate log file if it gets too large
def rotate_log_if_needed(log_file: Path, max_size_bytes: int = 10_000_000) -> None:  # 10MB default
    """
    Rotate the log file if it exceeds the maximum size.
    Creates a backup with timestamp and starts fresh.
    """
    if log_file.exists() and log_file.stat().st_size > max_size_bytes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = log_file.with_name(f"stderr_{timestamp}.log")
        log_file.rename(backup_file)
