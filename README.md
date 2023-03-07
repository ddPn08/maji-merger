# stable-diffusion-webui vae merger

A WebUI extension for merging VAE

# Usage

## Normal merge
![](./images/readme-normal-01.png)

|||
|-|-|
|VAE A, VAE B, VAE C|VAE used for merging|
|Output filename|File name for saving the merged VAE. It will be saved in `models/VAE` or the directory specified by `--vae-dir`.|
|Merge mode|Expression used for merging|
|Override|Overwrite the VAE if it already exists.|


## Each key merge
![](./images/readme-normal-02.png)

> **Note**
> I'm not sure if the implementation is correct or effective.

You can change the alpha of each key by moving the slider.