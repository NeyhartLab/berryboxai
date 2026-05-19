# berryboxai/shiny_app/run.py
import os
import sys
import subprocess
from pathlib import Path

def run_gui():
    """Entry point to launch the Shiny GUI."""
    # This finds the app.py file located in the same directory as this script
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)

    print(f"Launching BerryBox AI GUI from: {app_path}")
    
    # We call 'shiny run' as a module via the current python executable
    # This ensures it uses the same environment (conda/venv)
    try:
        subprocess.run([
            sys.executable, "-m", "shiny", "run", 
            "--launch-browser", 
            str(app_path)
        ])
    except KeyboardInterrupt:
        print("\nStopping BerryBox AI GUI...")