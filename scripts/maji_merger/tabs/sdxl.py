import gc
import json
import os
import gradio as gr

from modules import sd_hijack, devices, sd_models, shared

from scripts.maji_merger import merger, generator


def restore_ckpt():
    if shared.sd_model == None:
        return "already restored"
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    shared.sd_model = None
    gc.collect()
    devices.torch_gc()

    return "Restored checkpoint"


def merge_ckpt(ckpt_a, ckpt_b, ckpt_c, base_alpha, each_key, weights_txt, mode):
    ckpt_a = ckpt_a if type(ckpt_a) == str else None
    ckpt_b = ckpt_b if type(ckpt_b) == str else None
    ckpt_c = ckpt_c if type(ckpt_c) == str else None

    if each_key:
        weights = json.loads(weights_txt)
    else:
        weights = {}

    return "", merger.merge_sd_model(ckpt_a, ckpt_b, ckpt_c, base_alpha, weights, mode)


def merge_and_save(
    ckpt_a, ckpt_b, ckpt_c, base_alpha, each_key, weights_txt, mode, filename, override
):
    fullpath = merger.ckpt_fullpath(filename)
    assert not os.path.exists(fullpath) or override, "Checkpoint already exists"
    _, ckpt = merge_ckpt(
        ckpt_a, ckpt_b, ckpt_c, base_alpha, each_key, weights_txt, mode
    )
    merger.save_ckpt(ckpt, fullpath)

    return f"Merged checkpoint saved to {fullpath}"


def merge_and_generate(
    # merge options
    ckpt_a,
    ckpt_b,
    ckpt_c,
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
    _, ckpt = merge_ckpt(
        ckpt_a,
        ckpt_b,
        ckpt_c,
        base_alpha,
        each_key,
        weights_txt,
        mode,
    )

    generator.swap_sd_model(ckpt, ckpt_a)

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


def sdxl_tab():
    with gr.Tab("SDXL"):
        with gr.Column():
            with gr.Row():
                merge = gr.Button("Merge and save", variant="primary")
                merge_and_gen = gr.Button("Merge and gen", variant="primary")
            with gr.Row():
                restore = gr.Button("Restore Checkpoint")

            with gr.Row():
                ckpt_a = gr.Dropdown(
                    label="Checkpoint A",
                    choices=list(sd_models.checkpoint_tiles()),
                )
                ckpt_b = gr.Dropdown(
                    label="Checkpoint B",
                    choices=list(sd_models.checkpoint_tiles()),
                )
                ckpt_c = gr.Dropdown(
                    label="Checkpoint C",
                    choices=list(sd_models.checkpoint_tiles()),
                )
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
                    label="Base alpha",
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                )
                each_key = gr.Checkbox(False, label="Each key")
                override = gr.Checkbox(False, label="Override")

            weights_text = gr.TextArea(
                value=json.dumps(
                    {
                        "model.diffusion_model.input_blocks.0": 0.5,
                        "model.diffusion_model.input_blocks.1": 0.5,
                        "model.diffusion_model.input_blocks.2": 0.5,
                        "model.diffusion_model.input_blocks.3": 0.5,
                        "model.diffusion_model.input_blocks.4": 0.5,
                        "model.diffusion_model.input_blocks.5": 0.5,
                        "model.diffusion_model.input_blocks.6": 0.5,
                        "model.diffusion_model.input_blocks.7": 0.5,
                        "model.diffusion_model.input_blocks.8": 0.5,
                        "model.diffusion_model.middle_block": 0.5,
                        "model.diffusion_model.output_blocks.0": 0.5,
                        "model.diffusion_model.output_blocks.1": 0.5,
                        "model.diffusion_model.output_blocks.2": 0.5,
                        "model.diffusion_model.output_blocks.3": 0.5,
                        "model.diffusion_model.output_blocks.4": 0.5,
                        "model.diffusion_model.output_blocks.5": 0.5,
                        "model.diffusion_model.output_blocks.6": 0.5,
                        "model.diffusion_model.output_blocks.7": 0.5,
                        "model.diffusion_model.output_blocks.8": 0.5,
                    }
                ),
                visible=False,
            )

            with gr.Row(visible=False) as each_key_row:
                weights = {}
                with gr.Column():
                    for key in [
                        "model.diffusion_model.input_blocks.0",
                        "model.diffusion_model.input_blocks.1",
                        "model.diffusion_model.input_blocks.2",
                        "model.diffusion_model.input_blocks.3",
                        "model.diffusion_model.input_blocks.4",
                        "model.diffusion_model.input_blocks.5",
                        "model.diffusion_model.input_blocks.6",
                        "model.diffusion_model.input_blocks.7",
                        "model.diffusion_model.input_blocks.8",
                    ]:
                        weights[key] = gr.Slider(
                            label=key,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                        )
                with gr.Column():
                    for key in [
                        "model.diffusion_model.output_blocks.0",
                        "model.diffusion_model.output_blocks.1",
                        "model.diffusion_model.output_blocks.2",
                        "model.diffusion_model.output_blocks.3",
                        "model.diffusion_model.output_blocks.4",
                        "model.diffusion_model.output_blocks.5",
                        "model.diffusion_model.output_blocks.6",
                        "model.diffusion_model.output_blocks.7",
                        "model.diffusion_model.output_blocks.8",
                    ]:
                        weights[key] = gr.Slider(
                            label=key,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                        )

                with gr.Column():
                    for key in [
                        "model.diffusion_model.middle_block",
                    ]:
                        weights[key] = gr.Slider(
                            label=key,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
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
        ckpt_a,
        ckpt_b,
        ckpt_c,
        base_alpha,
        each_key,
        weights_text,
        mode,
    ]

    def initialize(status, gallery, geninfo, htmlinfo, htmllog, txt2img_preview_params):
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

        restore.click(fn=restore_ckpt, outputs=[status])

    return initialize
