import json
import os
import gradio as gr

from modules import sd_vae

from scripts.vae_merger import merger


def merge_vae(vae_a, vae_b, vae_c, base_alpha, weights_txt, mode, filename, override):
    fullpath = merger.vae_fullpath(filename)
    assert not os.path.exists(fullpath) or override, "VAE already exists"

    vae_a = vae_a if type(vae_a) != list else None
    vae_b = vae_b if type(vae_b) != list else None
    vae_c = vae_c if type(vae_c) != list else None

    weights = json.loads(weights_txt)
    vae = merger.each_key(vae_a, vae_b, vae_c, base_alpha, weights, mode)
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
            base_alpha = gr.Slider(
                label="Base alpha", minimum=0, maximum=1, value=0.5, step=0.01
            )
            with gr.Row():
                merge = gr.Button("Merge", variant="primary")
            with gr.Row():
                override = gr.Checkbox(False, label="Override")

        weights_text = gr.TextArea(
            value=json.dumps(
                {
                    "encoder.down.0": 0.5,
                    "encoder.down.1": 0.5,
                    "encoder.down.2": 0.5,
                    "encoder.down.3": 0.5,
                    "encoder.mid": 0.5,
                    "decoder.up.0": 0.5,
                    "decoder.up.1": 0.5,
                    "decoder.up.2": 0.5,
                    "decoder.up.3": 0.5,
                    "decoder.mid": 0.5,
                }
            )
        )

        with gr.Row():
            weights = {}
            with gr.Column():
                for key in [
                    "encoder.down.0",
                    "encoder.down.1",
                    "encoder.down.2",
                    "encoder.down.3",
                    "encoder.mid",
                ]:
                    weights[key] = gr.Slider(
                        label=key, minimum=0, maximum=1, step=0.01, value=0.5
                    )

            with gr.Column():
                for key in [
                    "decoder.up.0",
                    "decoder.up.1",
                    "decoder.up.2",
                    "decoder.up.3",
                    "decoder.mid",
                ]:
                    weights[key] = gr.Slider(
                        label=key, minimum=0, maximum=1, step=0.01, value=0.5
                    )

        def update_weights_text(data):
            d = {}
            for key in weights.keys():
                d[key] = data[weights[key]]
            return json.dumps(d)

        for w in weights.values():
            w.change(
                fn=update_weights_text,
                inputs={*weights.values()},
                outputs=[weights_text],
            )

        merge.click(
            fn=merge_vae,
            inputs=[
                vae_a,
                vae_b,
                vae_c,
                base_alpha,
                weights_text,
                mode,
                filename,
                override,
            ],
        )
