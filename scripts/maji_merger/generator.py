import torch

from modules import (
    processing,
    shared,
    sd_samplers,
    sd_models_xl,
    sd_hijack,
    sd_vae,
    images,
    ui,
    lowvram,
    devices,
)


def swap_sd_model(state_dict, original_name):
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    model = shared.sd_model

    model.is_sdxl = hasattr(model, "conditioner")
    model.is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, "model")
    model.is_sd1 = not model.is_sdxl and not model.is_sd2

    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)

    if not shared.cmd_opts.no_half:
        vae = model.first_stage_model

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None

        model.half()
        model.first_stage_model = vae

    devices.dtype = torch.float32 if shared.cmd_opts.no_half else torch.float16
    devices.dtype_vae = (
        torch.float32
        if shared.cmd_opts.no_half or shared.cmd_opts.no_half_vae
        else torch.float16
    )
    devices.dtype_unet = model.model.diffusion_model.dtype

    if hasattr(shared.cmd_opts, "upcast_sampling"):
        devices.unet_needs_upcast = (
            shared.cmd_opts.upcast_sampling
            and devices.dtype == torch.float16
            and devices.dtype_unet == torch.float16
        )
    else:
        devices.unet_needs_upcast = (
            devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
        )

    model.first_stage_model.to(devices.dtype_vae)
    sd_hijack.model_hijack.hijack(model)

    if hasattr(model, "logvar"):
        model.logvar = shared.sd_model.logvar.to(devices.device)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(model, shared.cmd_opts.medvram)
    else:
        model.to(shared.device)

    model.eval()

    shared.sd_model = model
    try:
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(
            force_reload=True
        )
    except:
        pass
    # shared.sd_model.sd_checkpoint_info.model_name = model_name

    def _setvae():
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(original_name)
        sd_vae.load_vae(shared.sd_model, vae_file, vae_source)

    try:
        _setvae()
    except:
        print("ERROR:setting VAE skipped")


def generate_img(prompt, nprompt, steps, sampler, cfg, seed, w, h):
    shared.state.begin()
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        do_not_save_grid=True,
        do_not_save_samples=True,
        do_not_reload_embeddings=True,
    )
    p.batch_size = 1
    p.prompt = prompt
    p.negative_prompt = nprompt
    p.steps = steps
    p.sampler_name = sd_samplers.samplers[sampler].name
    p.cfg_scale = cfg
    p.seed = seed
    p.width = w
    p.height = h
    p.seed_resize_from_w = 0
    p.seed_resize_from_h = 0
    p.denoising_strength = None

    if type(p.prompt) == list:
        p.all_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt
        ]
    else:
        p.all_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        ]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles)
            for x in p.negative_prompt
        ]
    else:
        p.all_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(
                p.negative_prompt, p.styles
            )
        ]

    processed: processing.Processed = processing.process_images(p)
    image = processed.images[0]

    infotext = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)
    if infotext.count("Steps: ") > 1:
        infotext = infotext[: infotext.rindex("Steps")]

    infotexts = infotext.split(",")
    infotext = ",".join(infotexts)
    images.save_image(
        image,
        shared.opts.outdir_txt2img_samples,
        "",
        p.seed,
        p.prompt,
        shared.opts.samples_format,
        p=p,
        info=infotext,
    )
    shared.state.end()
    return (
        processed.images,
        infotext,
        ui.plaintext_to_html(processed.info),
        ui.plaintext_to_html(processed.comments),
        p,
    )
