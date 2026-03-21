"""
streamlit_app.py  –  entry point for Streamlit Cloud / local launch.
All logic is in app.py; this file just invokes it.
"""
import os, sys

# Ensure the project root is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import main

main()
