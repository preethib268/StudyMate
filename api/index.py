import sys
from pathlib import Path

# Add the StudyMate app folder to Python's import path
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI
import gradio as gr

from studymate_app import demo

app = FastAPI()

app = gr.mount_gradio_app(
    app,
    demo,
    path="/"
)
