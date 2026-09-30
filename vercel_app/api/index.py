import sys
from pathlib import Path

APP_DIR = (
    Path(__file__).resolve().parent.parent
    / "vercel_app"
)

sys.path.insert(
    0,
    str(APP_DIR)
)

from fastapi import FastAPI
import gradio as gr

from app import demo


api = FastAPI()

api = gr.mount_gradio_app(
    api,
    demo,
    path="/"
)

app = api
