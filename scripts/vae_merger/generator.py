from modules import processing, shared, sd_samplers, images, ui


def generate_img(
    prompt,
    nprompt,
    steps,
    sampler,
    cfg,
    seed,
    w,
    h
):
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
