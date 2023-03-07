import os

from tqdm import tqdm
import torch
import safetensors.torch
from modules import sd_vae, shared


def vae_fullpath(filename):
    vae_path = (
        shared.cmd_opts.vae_dir
        if shared.cmd_opts.vae_dir is not None
        and os.path.isdir(shared.cmd_opts.vae_dir)
        else sd_vae.vae_path
    )
    return os.path.join(vae_path, filename)


def save_vae(vae, dest):
    print(f"Saving vae to {dest}")
    if dest.endswith(".safetensors"):
        safetensors.torch.save_file(vae, dest)
    else:
        torch.save(vae, dest)


def merge(vae_a, vae_b, vae_c, alpha, weights, mode):
    result = dict()

    print(f"Loading vae: {vae_a}")
    vae_a = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_a], map_location="cpu")
    print(f"Loading vae: {vae_b}")
    vae_b = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_b], map_location="cpu")
    if vae_c is not None:
        print(f"Loading vae: {vae_c}")
        vae_c = sd_vae.load_vae_dict(sd_vae.vae_dict[vae_c], map_location="cpu")

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

    vae_keys = (
        vae_a.keys() & vae_b.keys()
        if vae_c is None
        else vae_a.keys() & vae_b.keys() & vae_c.keys()
    )

    for key in tqdm(vae_keys):
        current_alpha = get_alpha(key) if weights is not None else alpha

        if mode == "Weight sum":
            result[key] = current_alpha * vae_a[key] + (1 - current_alpha) * vae_b[key]
        elif mode == "Add difference":
            assert vae_c is not None, "vae_c is undefined"
            result[key] = vae_a[key] + (vae_b[key] - vae_c[key]) * current_alpha

    return result
