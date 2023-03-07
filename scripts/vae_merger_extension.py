import gradio as gr

from modules import script_callbacks

from scripts.vae_merger import normal, each_key


def on_ui_tabs():
    with gr.Blocks() as block:
        with gr.Tabs():
            with gr.Tab(label="Normal"):
                normal.create_ui()
            with gr.Tab(label="Each key"):
                each_key.create_ui()

    return [(block, "VAE Merger", "vae_merger")]


script_callbacks.on_ui_tabs(on_ui_tabs)
