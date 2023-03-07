import os
import gradio as gr

from modules import sd_vae

from scripts.vae_merger import merger


def merge_vae(vae_a, vae_b, vae_c, alpha, mode, filename, override):
    fullpath = merger.vae_fullpath(filename)
    assert not os.path.exists(fullpath) or override, "VAE already exists"

    vae_a = vae_a if type(vae_a) != list else None
    vae_b = vae_b if type(vae_b) != list else None
    vae_c = vae_c if type(vae_c) != list else None

    vae = merger.normal_merge(vae_a, vae_b, vae_c, alpha, mode)
    merger.save_vae(vae, fullpath)


def create_ui():
    with gr.Column():
        with gr.Row():
            vae_a = gr.Dropdown(label="VAE A", choices=list(sd_vae.vae_dict.keys()))
            vae_b = gr.Dropdown(label="VAE B", choices=list(sd_vae.vae_dict.keys()))
            vae_c = gr.Dropdown(label="VAE C", choices=list(sd_vae.vae_dict.keys()))
            filename = gr.Text(label="Output filename")

        mode = gr.Radio(
            label="Merge mode",
            choices=[
                "Weight sum:A*(1-alpha)+B*alpha",
                "Add difference:A+(B-C)*alpha",
            ],
            value="Weight sum:A*(1-alpha)+B*alpha",
        )

        with gr.Row():
            alpha = gr.Slider(label="Alpha", minimum=0, maximum=1, value=0.5, step=0.01)
            with gr.Row():
                merge = gr.Button("Merge", variant="primary")
            with gr.Row():
                override = gr.Checkbox(False, label="Override")

        merge.click(
            fn=merge_vae, inputs=[vae_a, vae_b, vae_c, alpha, mode, filename, override]
        )
