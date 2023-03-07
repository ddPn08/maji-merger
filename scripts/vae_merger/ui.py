import argparse
import json
import os
import gradio as gr

from modules import sd_vae, ui, shared, script_callbacks

from scripts.vae_merger import merger, generator

txt2img_preview_params = None


def on_ui_train_tabs(params):
    global txt2img_preview_params
    txt2img_preview_params = params.txt2img_preview_params
    return None


script_callbacks.on_ui_train_tabs(on_ui_train_tabs)


def restore_vea():
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(shared.sd_model)
    sd_vae.load_vae(shared.sd_model, vae_file, vae_source)
    return "Restored VAE"


def merge_vae(vae_a, vae_b, vae_c, base_alpha, each_key, weights_txt, mode):
    vae_a = vae_a if type(vae_a) != list else None
    vae_b = vae_b if type(vae_b) != list else None
    vae_c = vae_c if type(vae_c) != list else None

    if each_key:
        weights = json.loads(weights_txt)
    else:
        weights = {}

    return "", merger.merge(vae_a, vae_b, vae_c, base_alpha, weights, mode)


def merge_and_save(
    vae_a, vae_b, vae_c, base_alpha, each_key, weights_txt, mode, filename, override
):
    fullpath = merger.vae_fullpath(filename)
    assert not os.path.exists(fullpath) or override, "VAE already exists"
    _, vae = merge_vae(vae_a, vae_b, vae_c, base_alpha, each_key, weights_txt, mode)
    merger.save_vae(vae, fullpath)

    return f"Merged VAE saved to {fullpath}"


def merge_and_generate(
    # merge options
    vae_a,
    vae_b,
    vae_c,
    base_alpha,
    each_key,
    weights_txt,
    mode,
    # txt2img options
    prompt,
    nprompt,
    steps,
    sampler,
    cfg,
    seed,
    w,
    h,
):
    _, vae = merge_vae(
        vae_a,
        vae_b,
        vae_c,
        base_alpha,
        each_key,
        weights_txt,
        mode,
    )

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    sd_vae._load_vae_dict(shared.sd_model, vae)

    return (
        "",
        *generator.generate_img(
            prompt,
            nprompt,
            steps,
            sampler,
            cfg,
            seed,
            w,
            h,
        ),
    )


def create_ui():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                merge = gr.Button("Merge and save", variant="primary")
                merge_and_gen = gr.Button("Merge and gen", variant="primary")
            with gr.Row():
                restore = gr.Button("Restore VAE")

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
                each_key = gr.Checkbox(False, label="Each key")
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
                ),
                visible=False,
            )

            with gr.Row(visible=False) as each_key_row:
                weights = {}
                with gr.Column():
                    for key in [
                        "encoder.conv_in",
                        "encoder.down.0",
                        "encoder.down.1",
                        "encoder.down.2",
                        "encoder.down.3",
                        "encoder.mid",
                        "encoder.norm_out",
                        "encoder.conv_out",
                        "quant_conv",
                    ]:
                        weights[key] = gr.Slider(
                            label=key, minimum=0, maximum=1, step=0.01, value=0.5
                        )

                with gr.Column():
                    for key in [
                        "decoder.conv_out",
                        "decoder.norm_out",
                        "decoder.up.0",
                        "decoder.up.1",
                        "decoder.up.2",
                        "decoder.up.3",
                        "decoder.mid",
                        "decoder.conv_in",
                        "post_quant_conv"
                    ]:
                        weights[key] = gr.Slider(
                            label=key, minimum=0, maximum=1, step=0.01, value=0.5
                        )
        with gr.Column():
            status = gr.Text(max_lines=1, show_label=False)
            gallery, geninfo, htmlinfo, htmllog = ui.create_output_panel(
                "txt2img", shared.opts.outdir_txt2img_samples
            )

        def each_key_on_change(each_key):
            return gr.update(visible=each_key), gr.update(visible=each_key)

        each_key.change(
            fn=each_key_on_change,
            inputs=[each_key],
            outputs=[weights_text, each_key_row],
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

        merge_data = [
            vae_a,
            vae_b,
            vae_c,
            base_alpha,
            each_key,
            weights_text,
            mode,
        ]

        merge.click(
            fn=merge_and_save,
            inputs=[
                *merge_data,
                filename,
                override,
            ],
            outputs=[status],
        )

        merge_and_gen.click(
            fn=merge_and_generate,
            inputs=[*merge_data, *txt2img_preview_params],
            outputs=[status, gallery, geninfo, htmlinfo, htmllog],
        )

        restore.click(fn=restore_vea, outputs=[status])
