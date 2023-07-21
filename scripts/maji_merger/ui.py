import gradio as gr

from modules import ui, shared, script_callbacks

from scripts.maji_merger.tabs import sd, sdxl, vae

txt2img_preview_params = None


def on_ui_train_tabs(params):
    global txt2img_preview_params
    txt2img_preview_params = params.txt2img_preview_params
    return None


script_callbacks.on_ui_train_tabs(on_ui_train_tabs)


def create_ui():
    with gr.Row():
        with gr.Tabs():
            init_sd = sd.sd_tab()
            init_sdxl = sdxl.sdxl_tab()
            init_vae = vae.vae_tab()
        with gr.Column():
            status = gr.Text(max_lines=1, show_label=False)
            gallery, geninfo, htmlinfo, htmllog = ui.create_output_panel(
                "txt2img", shared.opts.outdir_txt2img_samples
            )

    init_sd(status, gallery, geninfo, htmlinfo, htmllog, txt2img_preview_params)
    init_sdxl(status, gallery, geninfo, htmlinfo, htmllog, txt2img_preview_params)
    init_vae(status, gallery, geninfo, htmlinfo, htmllog, txt2img_preview_params)
