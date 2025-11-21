# src/__init__.py
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.append(str(src_path))