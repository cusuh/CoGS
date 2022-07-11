import math
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from main import instantiate_from_config
from cogs.modules.style.aladin import load_model


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self, transformer_config, image_encoder_config,
                 sketch_encoder_config, label_encoder_config,
                 permuter_config=None,
                 ckpt_path=None, ignore_keys=[],
                 image_key="image",
                 sketch_key="sketch",
                 style_key="cond_image",
                 label_key="label",
                 pkeep=1.0):

        super().__init__()
        self.init_image_encoder_from_ckpt(image_encoder_config)  # real images
        self.init_sketch_encoder_from_ckpt(sketch_encoder_config)  # sketches
        self.init_label_encoder_from_ckpt(label_encoder_config)  # labels
        if permuter_config is None:
            permuter_config = {"target": "cogs.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        self.aladin, self.resnet = load_model()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        self.sketch_key = sketch_key
        self.style_key = style_key
        self.label_key = label_key
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_image_encoder_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.image_encoder_model = model

    def init_sketch_encoder_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.sketch_encoder_model = model

    def init_label_encoder_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.label_encoder_model = model

    def forward(self, y, x, s, c):
        # one step to produce the logits
        quant_y, y_indices = self.encode_image(y)  # real image
        _, x_indices = self.encode_sketch(x)  # sketch
        _, s_indices = self.encode_image(s)  # style image
        _, c_indices = self.encode_label(c)  # label

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(
                self.pkeep * torch.ones(
                    y_indices.shape,device=y_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(
                y_indices, self.transformer.config.vocab_size)
            a_indices = mask*y_indices+(1-mask)*r_indices
        else:
            a_indices = y_indices

        cz_indices = torch.cat(
            (x_indices, s_indices, c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = y_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits_part = logits[:, x_indices.shape[1]+s_indices.shape[1]+c_indices.shape[1]-1:]

        return logits_part, target, quant_y, logits

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_image(self, x):
        quant_z, _, info = self.image_encoder_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_sketch(self, x):
        quant_z, _, info = self.sketch_encoder_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_label(self, x):
        quant_z, _, info = self.label_encoder_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.image_encoder_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.image_encoder_model.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_img_from_logits(self, logits, zshape,
                               temperature=1.0, top_k=100, sample=True):
        x = []
        for logit_ind in range(0, logits.shape[1]):
            logits_i = logits[:, logit_ind, :] / temperature
            if top_k is not None:
                logits_i = self.top_k_logits(logits_i, top_k)
            probs = F.softmax(logits_i, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, ix), 1) if len(x) else ix

        index = x[:, -zshape[1]:]

        x = self.decode_to_img(index, zshape)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            y, x, s, c = self.prepare_batch(batch, N, diffuse=False, upsample_factor=8)
        else:
            y, x, s, c = self.prepare_batch(batch, N)
        y = y.to(device=self.device)
        x = x.to(device=self.device)
        s = s.to(device=self.device)
        c = c.to(device=self.device)

        quant_y, y_indices = self.encode_image(y)  # real image
        quant_x, x_indices = self.encode_sketch(x)  # sketch
        quant_s, s_indices = self.encode_image(s)  # style
        quant_c, c_indices = self.encode_label(c)  # label

        cond_indices = torch.cat((x_indices, s_indices, c_indices), dim=1)

        # create a "half" sample
        y_start_indices = y_indices[:,:y_indices.shape[1]//2]
        index_sample = self.sample(y_start_indices, cond_indices,
                                   steps=y_indices.shape[1]-y_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        y_sample = self.decode_to_img(index_sample, quant_y.shape)

        # sample
        y_start_indices = y_indices[:, :0]
        index_sample = self.sample(y_start_indices, cond_indices,
                                   steps=y_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        y_sample_nopix = self.decode_to_img(index_sample, quant_y.shape)

        # det sample
        y_start_indices = y_indices[:, :0]
        index_sample = self.sample(y_start_indices, cond_indices,
                                   steps=y_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        y_sample_det = self.decode_to_img(index_sample, quant_y.shape)

        # reconstruction
        y_rec = self.decode_to_img(y_indices, quant_y.shape)

        log["inputs"] = y
        log["reconstructions"] = y_rec

        x_rec = self.sketch_encoder_model.decode(quant_x)
        log["sketch_rec"] = x_rec
        log["sketch"] = x
        
        s_rec = self.decode_to_img(s_indices, quant_s.shape)
        log["style_rec"] = s_rec
        log["style"] = s

        log["samples_half"] = y_sample
        log["samples_nopix"] = y_sample_nopix
        log["samples_det"] = y_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def prepare_batch(self, batch, N=None):
        y = self.get_input(self.image_key, batch)
        x = self.get_input(self.sketch_key, batch)
        s = self.get_input(self.style_key, batch)
        c = batch[self.label_key]
        if N is not None:
            y = y[:N]
            x = x[:N]
            s = s[:N]
            c = c[:N]
        return y, x, s, c

    def aladin_forward(self, image):
        transform_fn = transforms.Compose([
            transforms.Resize(256),
        ])
        image = transform_fn(image)
        image = torch.clamp(image, -1., 1.)

        embedding_aladin = self.aladin.encode(image, just_style_code=True)
        embedding_aladin = self.aladin.proj_head(embedding_aladin)
        embedding_aladin = embedding_aladin / embedding_aladin.norm(dim=1)[:, None]

        embedding_resnet = self.resnet(image)
        embedding_resnet = embedding_resnet / embedding_resnet.norm(dim=1)[:, None]
        embedding_resnet = embedding_resnet.view((-1, embedding_resnet.shape[1]))

        embedding = torch.cat([embedding_aladin, embedding_resnet], dim=1)
        return embedding

    def shared_step(self, batch, batch_idx):
        y, x, s, c = self.prepare_batch(batch)
        logits, target, quant_y, logits_full = self(y, x, s, c)
        pred_feat = self.aladin_forward(self.decode_img_from_logits(logits_full, quant_y.shape))
        gt_feat = self.aladin_forward(s)
        style_loss = F.mse_loss(pred_feat, gt_feat)
        codebook_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss = codebook_loss + style_loss
        return total_loss, codebook_loss, style_loss

    def training_step(self, batch, batch_idx):
        loss, c_loss, s_loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/xe_loss', c_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/mse_loss', s_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, c_loss, s_loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/xe_loss', c_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/mse_loss', s_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
