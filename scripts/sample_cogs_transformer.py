import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data.dataloader import default_collate

from main import instantiate_from_config
from cogs.modules.transformer.mingpt import sample_with_past


rescale = lambda x: (x + 1.) / 2.


def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


@torch.no_grad()
def decode_to_img(index, zshape):
    index = model.permuter(index, reverse=True)
    bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
    quant_z = model.first_stage_model.quantize.get_codebook_entry(
        index.reshape(-1), shape=bhwc)
    x = model.first_stage_model.decode(quant_z)
    return x


@torch.no_grad()
def sample_conditional(model, dsets, split='validation', verbose_time=False):
    print(f'DSET KEYS: {sorted(dsets.datasets.keys())}')
    if len(dsets.datasets) > 1:
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    batch_size = 1
    temperature = 1.0
    top_k = 100
    top_p = 1.0
    sample = True
    callback = None

    for start_index in range(len(dset)):
        indices = list(range(start_index, start_index+batch_size))

        example = default_collate([dset[i] for i in indices])

        y, x, s, c = model.prepare_batch(example)
        y = y.to(device=model.device)
        x = x.to(device=model.device)
        s = s.to(device=model.device)
        c = c.to(device=model.device)

        quant_y, y_indices = model.encode_image(y)
        quant_x, x_indices = model.encode_sketch(x)
        quant_s, s_indices = model.encode_image(s)
        quant_c, c_indices = model.encode_label(c)

        cond_indices = torch.cat((x_indices, s_indices, c_indices), dim=1)

        print(f'{start_index}/{len(dset)}')
        t1 = time.time()
        index_sample = sample_with_past(cond_indices, model.transformer, steps=y_indices.shape[1],
                                        sample_logits=sample, top_k=top_k, callback=callback,
                                        temperature=temperature, top_p=top_p)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        y_sample_nopix = model.decode_to_img(index_sample, quant_y.shape)

        out_dir = f'{opt.out_dir}/{opt.resume}/'
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        chw_to_pillow(y.squeeze(0)).save(f'{out_dir}/gt-{start_index}.png')
        chw_to_pillow(x.squeeze(0)).save(f'{out_dir}/sketch-{start_index}.png')
        chw_to_pillow(s.squeeze(0)).save(f'{out_dir}/style-{start_index}.png')
        chw_to_pillow(y_sample_nopix.squeeze(0)).save(f'{out_dir}/sample-{start_index}.png')


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="name of directory to save outputs",
        default="samples")
    parser.add_argument(
        "--split",
        type=str,
        help="which dataset split to generate samples from [train, validation]",
        default="validation")
    return parser


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    print(f'Resume: {opt.resume}')
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    print(f'Checkpoint: {ckpt}')

    gpu = True
    eval_mode = True

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    print(f"Global step: {global_step}")
    sample_conditional(model, dsets, opt.split)
