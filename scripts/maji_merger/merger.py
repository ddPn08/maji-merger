import os

from tqdm import tqdm
import torch
import safetensors.torch
from modules import sd_models, sd_vae, shared


def ckpt_fullpath(filename):
    ckpt_path = (
        shared.cmd_opts.ckpt_dir
        if shared.cmd_opts.ckpt_dir is not None
        and os.path.isdir(shared.cmd_opts.ckpt_dir)
        else sd_models.model_dir
    )
    return os.path.join(ckpt_path, filename)


def vae_fullpath(filename):
    vae_path = (
        shared.cmd_opts.vae_dir
        if shared.cmd_opts.vae_dir is not None
        and os.path.isdir(shared.cmd_opts.vae_dir)
        else sd_vae.vae_path
    )
    return os.path.join(vae_path, filename)


def save_ckpt(ckpt, dest):
    print(f"Saving ckpt to {dest}")
    if dest.endswith(".safetensors"):
        safetensors.torch.save_file(ckpt, dest)
    else:
        torch.save(ckpt, dest)


def load_model(model):
    info = sd_models.get_closet_checkpoint_match(model)
    name = info.model_name

    if info in sd_models.checkpoints_loaded:
        print(f"Loading model {name} from cache")
        return sd_models.checkpoints_loaded[info]
    elif shared.opts.sd_checkpoint_cache > 0:
        sd_models.load_model(info)
        print(f"Loading model {name} from cache")
        return sd_models.checkpoints_loaded[info]
    else:
        sd_models.load_model(info)
        print(f"Loading model {name} from file")
        try:
            return sd_models.read_state_dict(info.filename, "cuda")
        except:
            return sd_models.read_state_dict(info.filename)


def merge_state_dict(sd_a, sd_b, sd_c, alpha, weights, mode):
    result = dict()

    def get_alpha(key):
        try:
            filtered = sorted(
                [x for x in weights.keys() if key.startswith(x)], key=len, reverse=True
            )
            if len(filtered) < 1:
                return alpha
            return weights[filtered[0]]
        except:
            return alpha

    mode, *_ = mode.split(":")

    ckpt_keys = (
        sd_a.keys() & sd_b.keys()
        if sd_c is None
        else sd_a.keys() & sd_b.keys() & sd_c.keys()
    )

    for key in tqdm(ckpt_keys):
        current_alpha = get_alpha(key) if weights is not None else alpha

        if mode == "Weight sum":
            result[key] = (1 - current_alpha) * sd_a[key] + current_alpha * sd_b[key]
        elif mode == "Add difference":
            assert sd_c is not None, "ckpt_c is undefined"
            result[key] = sd_a[key] + (sd_b[key] - sd_c[key]) * current_alpha

    return result


def merge_sd_model(ckpt_a_name, ckpt_b_name, ckpt_c_name, alpha, weights, mode):
    print(f"Loading ckpt: {ckpt_a_name}")
    ckpt_a = load_model(ckpt_a_name)
    print(f"Loading ckpt: {ckpt_b_name}")
    ckpt_b = load_model(ckpt_b_name)
    ckpt_c = None
    if ckpt_c_name:
        print(f"Loading ckpt: {ckpt_c_name}")
        ckpt_c = load_model(ckpt_c_name)

    return merge_state_dict(ckpt_a, ckpt_b, ckpt_c, alpha, weights, mode)


def merge_vae(vae_a_path, vae_b_path, vae_c_path, alpha, weights, mode):
    print(f"Loading vae: {vae_a_path}")
    vae_a = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_a_path], map_location="cpu")
    print(f"Loading vae: {vae_b_path}")
    vae_b = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_b_path], map_location="cpu")
    vae_c = None
    if vae_c_path in sd_vae.vae_dict:
        print(f"Loading vae: {vae_c_path}")
        vae_c = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_c_path], map_location="cpu")

    return merge_state_dict(vae_a, vae_b, vae_c, alpha, weights, mode)
