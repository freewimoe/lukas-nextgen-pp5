#!/usr/bin/env python3
"""
Simple runner for LUKAS NextGen Streamlit app.
This solves all import path issues by ensuring correct Python path setup.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Now we can safely import and run the app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Set up streamlit arguments
    sys.argv = ["streamlit", "run", "app/app.py"]
    stcli.main()
