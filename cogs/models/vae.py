import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import numpy as np
from torchvision import transforms as T
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image

from main import instantiate_from_config
from cogs.modules.vae.model import vEncoderSE, DecoderSE


class ContrastiveVAE(pl.LightningModule):
    def __init__(self,
                 vaeconfig,
                 lossconfig,
                 output_name=None,
                 ckpt_path=None,
                 cogs_transformer_config=None,
                 ignore_keys=[],
                 supcon_epochs=50,
                 image_key="image",
                 sketch_key="sketch",
                 style_key="cond_image",
                 label_key="label"
                 ):
        super().__init__()
        self.init_transformer_from_ckpt(cogs_transformer_config)
        self.vqgan_batch = 16
        self.encoder = vEncoderSE(**vaeconfig)
        self.decoder = DecoderSE(**vaeconfig)
        self.loss = instantiate_from_config(lossconfig)
        if ckpt_path is not None:
            self.output_name = output_name
            self.init_ae_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.supcon_epochs = supcon_epochs
        self.reconstruction_loss = False
        self.image_key = image_key
        self.sketch_key = sketch_key
        self.style_key = style_key
        self.label_key = label_key


    def init_transformer_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.transformer = model

    def init_ae_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        os.makedirs(f'output/{self.output_name}', exist_ok=True)
        print(f'Created directory to save outputs: output/{self.output_name}')

    def encode(self, x):
        h_mean, h_std = self.encoder(x)
        return h_mean, h_std

    def decode(self, x):
        h = self.decoder(x)
        return h

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar).to('cuda:0')
        q = torch.distributions.Normal(mean, stddev)
        z = q.rsample()
        return z

    def forward(self, y, x, s, c):

        quant_y, y_indices = self.transformer.encode_image(y) #real image
        _, x_indices = self.transformer.encode_sketch(x) #sketch
        _, s_indices = self.transformer.encode_image(s) #style image
        _, c_indices = self.transformer.encode_label(c) #label

        zshape = quant_y.shape
        a_indices = y_indices

        cz_indices = torch.cat((x_indices, s_indices, c_indices, a_indices), dim=1)

        # make the prediction
        logits, _ = self.transformer.transformer(cz_indices[:, :-1])

        temperature = 1.0
        x = []
        for logit_ind in range(0, logits.shape[1]):
            logits_i = logits[:, logit_ind, :] / temperature
            # optionally crop probabilities to only the top k options
            top_k = 100
            if top_k is not None:
                logits_i = self.transformer.top_k_logits(logits_i, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits_i, dim=-1)
            # sample from the distribution or take the most likely
            sample = True
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, ix), 1) if len(x) else ix

        index = x[:, x_indices.shape[1] + s_indices.shape[1] + c_indices.shape[1] - 1:]
        index = self.transformer.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.transformer.image_encoder_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)

        z = quant_z

        enc_mean, enc_std = self.encode(z)
        enc = self.sample_z(enc_mean, enc_std)
        noise = Variable(torch.FloatTensor(x.shape[0], 1024).normal_(0, 1)).to('cuda:0')
        dec = self.decode(torch.cat((enc, noise), dim=1))
        return dec, enc, z, enc_mean, enc_std


    def get_style_images(self, image_path, batch):
        s1 = batch[image_path]
        s2 = s1[torch.randperm(s1.shape[0]), ...]

        if len(s1.shape) == 3:
            s1 = s1[..., None]
        s1 = s1.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if s1.dtype == torch.double:
            s1 = s1.float()

        if len(s2.shape) == 3:
            s2 = s2[..., None]
        s2 = s2.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if s2.dtype == torch.double:
            s2 = s2.float()

        return s1, s2

    def prepare_batch(self, batch, N=None):
        y = self.transformer.get_input(self.transformer.image_key, batch)
        x = self.transformer.get_input(self.transformer.sketch_key, batch)
        s1, s2 = self.get_style_images(self.transformer.style_key, batch)
        c = batch[self.transformer.label_key]

        if N is not None:
            y = y[:N]
            x = x[:N]
            s1 = s1[:N]
            s2 = s2[:N]
            c = c[:N]
        return y, x, s1, s2, c

    def get_input(self, batch, split):
        """
        For each sample in the batch, we create and save two views, obtained through data augmentation, obtaining a
        multiview batch that has the double of samples
        """
        y, x, s1, s2, c = self.prepare_batch(batch)

        if split != 'test':
            y_out_1 = []
            y_out_2 = []
            x_out_1 = []
            x_out_2 = []
            s_1 = []
            s_2 = []
            for i, y_i in enumerate(y):
                y_i1 = y_i
                x_i1 = x[i]
                s_i1 = s1[i]
                y_i2 = y_i
                x_i2 = x[i]
                s_i2 = s2[i]
                y_out_1 = torch.cat((y_out_1, torch.unsqueeze(y_i1, 0)), 0) if len(y_out_1) else torch.unsqueeze(y_i1, 0)
                y_out_2 = torch.cat((y_out_2, torch.unsqueeze(y_i2, 0)), 0) if len(y_out_2) else torch.unsqueeze(y_i2, 0)

                x_out_1 = torch.cat((x_out_1, torch.unsqueeze(x_i1, 0)), 0) if len(x_out_1) else torch.unsqueeze(x_i1,                                                                                                                0)
                x_out_2 = torch.cat((x_out_2, torch.unsqueeze(x_i2, 0)), 0) if len(x_out_2) else torch.unsqueeze(x_i2,
                                                                                                                 0)

                s_1 = torch.cat((s_1, torch.unsqueeze(s_i1, 0)), 0) if len(s_1) else torch.unsqueeze(s_i1, 0)
                s_2 = torch.cat((s_2, torch.unsqueeze(s_i2, 0)), 0) if len(s_2) else torch.unsqueeze(s_i2, 0)

            y = torch.cat((y_out_1, y_out_2), 0)
            x = torch.cat((x_out_1, x_out_2), 0)
            s = torch.cat((s_1, s_2), 0)
            c = torch.cat((c, c), 0)
        else:
            s = s1

        return y, x, s, c

    def training_step(self, batch, batch_idx):
        y, x, s, c = self.get_input(batch, split='train')
        z_rec, enc_ae, z_vqgan, enc_mean, enc_std = self(y, x, s, c)

        enc_ae_split = F.normalize(enc_ae, p=2, dim=1)

        f1, f2 = torch.split(enc_ae_split, [int(enc_ae_split.shape[0]/2), int(enc_ae_split.shape[0]/2)], dim=0)
        enc_ae_split = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if self.current_epoch > self.supcon_epochs:
            self.reconstruction_loss = True
        loss, log_losses = self.loss(z_vqgan, z_rec, enc_ae_split, enc_ae, enc_mean, enc_std,
                                     split='train', recons=self.reconstruction_loss)  # needs labels, have to see data format, where are the labels

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_losses, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, x, s, c = self.get_input(batch, split='val')
        z_rec, enc_ae, z_vqgan, enc_mean, enc_std = self(y, x, s, c)

        enc_ae_split = F.normalize(enc_ae, p=2, dim=1)
        f1, f2 = torch.split(enc_ae_split, [int(enc_ae_split.shape[0]/2), int(enc_ae_split.shape[0]/2)], dim=0)
        enc_ae_split = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if self.current_epoch > self.supcon_epochs:
            self.reconstruction_loss = True
        loss, log_losses = self.loss(z_vqgan, z_rec, enc_ae_split, enc_ae, enc_mean, enc_std,
                                     split='val', recons=self.reconstruction_loss)  # needs labels, have to see data format, where are the labels

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_losses, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict


    def to_RGB(self, x):
        x = x.detach().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.) / 2.

        return x

    def plot_tsne_images(self, points, images, lr=100, metric='cosine', name='all'):

        tsne = TSNE(n_components=2, learning_rate=lr, metric=metric)
        X = tsne.fit_transform(points)
        print('points fitted')

        tx, ty = X[:, 0], X[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 4000
        height = 3000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(images, tx, ty):
            img = torch.as_tensor(img)
            img = self.to_RGB(img)
            tile = T.ToPILImage()(img.squeeze_(0))
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

        plt.figure(figsize=(16, 12))

        full_image.save(f'output/{self.output_name}/tsne_images_{name}.png')

    def compute_knn(self, x, k):
        scores = cosine_similarity(torch.tensor(x))
        _, pred_indices = torch.topk(torch.tensor(scores), k + 1)
        return pred_indices


    def sample_img(self, x, c, steps, temperature=1.0, sample=False, top_k=None):
        x = torch.cat((c, x), dim=1)
        block_size = self.transformer.transformer.get_block_size()
        assert not self.transformer.transformer.training
        if self.transformer.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape) == 2
            noise_shape = (x.shape[0], steps - 1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:, x.shape[1] - c.shape[1]:-1]
            x = torch.cat((x, noise), dim=1)
            logits, _ = self.transformer.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.transformer.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0], shape[1], shape[2])
                ix = ix.reshape(shape[0], shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1] - 1:]
        else:
            for k in range(steps):
                assert x.size(1) <= block_size  # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.transformer.top_k_logits(logits, top_k)
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


    def decode_to_vae(self, index, zshape):
        index = self.transformer.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.transformer.image_encoder_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        return quant_z

    def test_step(self, batch, batch_idx):

        y, x, s, c = self.get_input(batch, split='test')

        quant_y, y_indices = self.transformer.encode_image(y) #real image
        quant_x, x_indices = self.transformer.encode_sketch(x) #sketch
        quant_s, s_indices = self.transformer.encode_image(s) #style image
        quant_c, c_indices = self.transformer.encode_label(c) #label

        xsc_indices = torch.cat((x_indices, s_indices, c_indices), dim=1)

        temperature = 1.0  # st.number_input("Temperature", value=1.0)
        top_k = 100  # st.number_input("Top k", value=100)

        # sample
        y_start_indices = y_indices[:, :0]
        index_sample = self.sample_img(y_start_indices, xsc_indices,
                                  steps=y_indices.shape[1],
                                  temperature=temperature if temperature is not None else 1.0,
                                  sample=True,
                                  top_k=top_k if top_k is not None else 100)
        z = self.decode_to_vae(index_sample, quant_y.shape)

        enc_mean, enc_std = self.encode(z)
        enc_ae = self.sample_z(enc_mean, enc_std)

        prevae = self.transformer.image_encoder_model.decode(z)

        noise = Variable(torch.FloatTensor(x.shape[0], 1024).normal_(0, 1)).to('cuda:0')
        vae_out = self.decode(torch.cat((enc_ae, noise), dim=1))
        postvae = self.transformer.image_encoder_model.decode(vae_out)
        output_img = self.to_RGB(postvae)
        save_image(output_img, f'output/{self.output_name}/vae_img_{batch_idx}.png')
        print(f'Image output/{self.output_name}/vae_img_{batch_idx}.png saved!')

        return enc_ae, prevae

    def test_epoch_end(self, encodings):

        # Save t-sne plot for all embeddings

        points = []
        images = []
        for encoding in encodings:
            if len(points):
                points.extend(encoding[0].tolist())
                images.extend(encoding[1].tolist())
            else:
                points = encoding[0].tolist()
                images = encoding[1].tolist()

        self.plot_tsne_images(points, images, name='all_cosine')

        # Find top 5 retrieval
        index_closest_5 = self.compute_knn(points, 5)

        #Retrieve top 5 images
        for n_1, image_indices in enumerate(index_closest_5):
            if not n_1 % 10:
                print(f'Retrieving images for {n_1}')
                retrieval_main_image = self.to_RGB(torch.as_tensor(images[n_1]))
                save_image(retrieval_main_image, f'output/{self.output_name}/retrieval_{n_1}.png')
                for n_2, index in enumerate(image_indices):
                    print(f'Retrieved image number {index}')
                    retrieved_image = self.to_RGB(torch.as_tensor(images[index]))
                    save_image(retrieved_image, f'output/{self.output_name}/retrieval_{n_1}_{n_2}.png')


    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                               list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))


        return opt




