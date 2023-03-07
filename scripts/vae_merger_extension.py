import gradio as gr

from modules import script_callbacks

from scripts.vae_merger import ui


def on_ui_tabs():
    with gr.Blocks() as block:
        ui.create_ui()

    return [(block, "VAE Merger", "vae_merger")]


script_callbacks.on_ui_tabs(on_ui_tabs)
