import subprocess
import sys
from pathlib import Path

def open_native_dialog(title="Select folder"):
    """
    Standard Python function to open a folder picker.
    Works independently of Streamlit or Shiny.
    """
    script = (
        f"import tkinter as tk; from tkinter import filedialog; "
        f"root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1); "
        f"print(filedialog.askdirectory(title={repr(title)})); root.destroy()"
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except Exception:
        return ""