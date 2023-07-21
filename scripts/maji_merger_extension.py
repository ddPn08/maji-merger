import gradio as gr

from modules import script_callbacks

from scripts.maji_merger import ui


def on_ui_tabs():
    with gr.Blocks() as block:
        ui.create_ui()

    return [(block, "Maji Merger", "maji_merger")]


script_callbacks.on_ui_tabs(on_ui_tabs)
