import argparse
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

from cogs.modules.style.hparams import initParams


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class AdaIN (torch.nn.Module):

    def __init__ (self, hparams):
        super(AdaIN, self).__init__()

        self.hparams = hparams

    def forward(self, f_vol):
        batch = f_vol.view(f_vol.shape[0], f_vol.shape[1], f_vol.shape[2]*f_vol.shape[3])

        std = batch.std(2)
        std_0 = torch.min(std)==0
        while std_0: # Gradient of std 0 leads to NaN
            noise = batch.new(batch).normal_(0, 1e-8)
            batch = batch + noise
            std = batch.std(2)
            std_0 = torch.min(std)==0

        if self.hparams.styleGANmod:
            adain_vals = std
        else:
            means = batch.mean(2)
            adain_vals = torch.cat([means, std], dim=-1)
        return adain_vals


class VGG_Dec(nn.Module):

    def __init__ (self, hparams, enc_style):
        super(VGG_Dec, self).__init__()
        self.hparams = hparams
        self.enc_style = enc_style

        self.enc_layers = list(self.enc_style.model.named_children())
        self.enc_layers.reverse()

        self.layers = []
        conv_counter = 0
        pool_counter = 0
        self.filter_per_layer = []
        for name, e_layer in self.enc_layers:
            if isinstance(e_layer, nn.Conv2d):
                enc_out = e_layer.weight.shape[0]
                enc_in = e_layer.in_channels
                self.filter_per_layer.append(enc_in)
                self.layers.append((f'conv_{conv_counter}', nn.Conv2d(enc_out, enc_in, kernel_size=e_layer.kernel_size, padding=e_layer.padding)))
                self.layers.append((f'ain_{conv_counter}', AdaptiveInstanceNorm2d(enc_in) ))
                self.layers.append((f'relu_{conv_counter}', nn.ReLU(inplace=True)))
                conv_counter += 1

            elif isinstance(e_layer, nn.MaxPool2d):
                self.layers.append((f'unpool_{pool_counter}', nn.UpsamplingNearest2d(scale_factor=2)))
                pool_counter += 1
                pass

        self.filter_per_layer.reverse()
        self.model = nn.Sequential(OrderedDict(self.layers))


    def apply_adain(self, adain):
        layer_params = []

        # Split up the AdaIN style code into the mean/std values for each layer
        start = 0
        for p in range(len(self.filter_per_layer)):
            ppl = self.filter_per_layer[p]
            this_layer_params = adain[:, start:start+ppl*2]
            mean, std = this_layer_params[:, 0:int(this_layer_params.shape[1]/2)], this_layer_params[:, int(this_layer_params.shape[1]/2):]
            layer_params.append([mean, std])
            start += ppl*2
            del this_layer_params

        layer_params.reverse()

        params_i = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                [mean, std] = layer_params[params_i]

                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)

                params_i += 1


        del layer_params

    def forward(self, x):
        return self.model(x)


class VGG_AdaIN(nn.Module):
    def __init__(self, hparams, encoder_layers=None, also_content=False):
        super(VGG_AdaIN, self).__init__()

        self.hparams = hparams
        self.is_content = False
        self.also_content = also_content

        size_multipliers = {}
        size_multipliers["small"] = 1
        size_multipliers["wide2x"] = 2
        size_multipliers["wide3x"] = 3
        size_multipliers["wide4x"] = 4
        size_multipliers["wide8x"] = 8
        self.width = size_multipliers[self.hparams.size] if self.hparams.size in list(size_multipliers.keys()) else 1
        self.adaIN_size = 0

        architecture = "vgg16" if "16" in self.hparams.backbone else "vgg19"
        self.vgg_model = getattr(models, architecture)(pretrained=self.hparams.pretrained)

        layers = list(self.vgg_model.named_children())[0][1]
        self.encoder_layers = []
        for l, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                if self.width>1:
                    in_channels = layer.in_channels if l==0 else layer.in_channels*self.width
                    layer = nn.Conv2d(in_channels, layer.weight.shape[0]*self.width, kernel_size=layer.kernel_size, padding=layer.padding)
                self.encoder_layers.append(("c", layer))
                self.adaIN_size += 2*layer.weight.shape[0]

            elif isinstance(layer, nn.MaxPool2d):
                self.encoder_layers.append(("p", layer))
        del self.vgg_model
        self.model = []

        self.num_layers = len(self.encoder_layers)

        # For computing FC layers sizes
        last_channels = 3
        last_dim = self.hparams.resize_shape

        conv_count = 0
        pool_count = 0
        for l in range(len(self.encoder_layers)):

            [l_type, layer] = self.encoder_layers[l]

            if l_type=="c":
                self.model.append((f'conv_{conv_count}', layer))

                self.model.append((f'adain_{conv_count}', AdaIN(hparams) ))
                conv_count += 1

            elif l_type=="p":
                self.model.append((f'pool_{pool_count}', layer))
                last_dim = int(last_dim/layer.stride)
                pool_count += 1

        self.model.append(("flatten", Flatten()))

        self.model = nn.Sequential(OrderedDict(self.model))

        self.layers_list = self.model.named_children()
        self.layers = dict(self.model.named_children())
        self.relu = F.leaky_relu

    def forward(self, data):
        self.layers = dict(self.model.named_children())

        latent = []
        cat_dim = 1

        conv_count = 0
        pool_count = 0
        in_count = 0
        for l in range(self.num_layers):

            l_type = self.encoder_layers[l][0]

            if l_type == "c":

                feat = self.relu( self.layers[f'conv_{conv_count}'](data) )
                if not self.is_content:
                    if self.hparams.latent_mode == "fc":
                        cat_dim = 1
                        latent.append(self.layers[f'linear_{conv_count}']( self.layers["flatten"](feat) ))
                    if self.hparams.latent_mode == "adain":
                        latent.append(self.layers[f'adain_{conv_count}']( feat ))

                data = feat
                conv_count += 1
            elif l_type == "p":
                data = self.layers[f'pool_{pool_count}'](data)
                pool_count += 1

            elif l_type == "in":
                data = self.layers[f'in_{in_count}'](data)
                in_count += 1

        if self.is_content:
            return data
        latent = torch.cat(latent, dim=cat_dim)

        if self.also_content:
            return data, latent

        del data
        return latent

    def get_feats(self, image):
        feats = []
        conv_count = 0
        pool_count = 0
        in_count = 0
        for l in range(self.num_layers):
            l_type = self.encoder_layers[l][0]
            if l_type == "c":
                feat = self.relu( self.layers[f'conv_{conv_count}'](image) )
                feats.append(feat)
                conv_count += 1
                image = feat
            elif l_type == "p":
                image = self.layers[f'pool_{pool_count}'](image)
                pool_count += 1

            elif l_type == "in":
                image = self.layers[f'in_{in_count}'](image)
                in_count += 1
        return feats


    def get_feat_map_adain(self, f_map):
        f_map = f_map[0] # TODO, implement batch size, instead of sample size

        means = []
        stds = []

        no_channels = f_map.shape[0]

        for c in range(no_channels):
            means.append(torch.mean(f_map[c]))
            stds.append(torch.std(f_map[c]))

        adain = means + stds
        adain = torch.cat([v.view(1) for v in adain])

        return adain

    # https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
    def get_fc_size(self, conv, channels, dim):
        s = conv.stride[0]
        p = conv.padding[0]
        k = conv.kernel_size[0]
        i = dim

        out_size = (i-k+2*p)/s + 1
        return conv(torch.rand(1, channels, dim, dim)).data.view(1, -1).size(1), int(out_size)

class AVGG(nn.Module):

    def __init__(self, hparams):
        super(AVGG, self).__init__()

        self.hparams = hparams

        size_multipliers = {}
        size_multipliers["small"] = 1
        size_multipliers["wide2x"] = 2
        size_multipliers["wide3x"] = 3
        size_multipliers["wide4x"] = 4
        size_multipliers["wide8x"] = 8
        self.width = size_multipliers[self.hparams.size] if self.hparams.size in list(size_multipliers.keys()) else 1

        self.enc_style = VGG_AdaIN(hparams)
        self.enc_content = VGG_AdaIN(hparams)
        self.enc_content.is_content = True
        self.dec = VGG_Dec(hparams, self.enc_style)

        if self.hparams.proj_head:
            layers = []
            layers.append(("relu1", nn.LeakyReLU()))
            layers.append(("fc1", nn.Linear(self.enc_style.adaIN_size, 1024)))
            layers.append(("relu2", nn.LeakyReLU()))
            layers.append(("fc2", nn.Linear(1024, 256)))
            self.proj_head = nn.Sequential(OrderedDict(layers))

        if self.hparams.use_style_mlp:
            layers = []
            in_size = 8448 if "16" in self.hparams.backbone else 11008 # VGG16/19
            in_size *= self.width
            mid_size = int(self.hparams.style_dim + (in_size - self.hparams.style_dim) / 2)
            layers.append(("style_mlp_1", nn.Linear(in_size, mid_size)))
            layers.append(("style_mlp_r", nn.LeakyReLU()))
            layers.append(("style_mlp_2", nn.Linear(mid_size, self.hparams.style_dim)))
            self.style_mlp = nn.Sequential(OrderedDict(layers))

        # Network weight initialization
        self.apply(self.weights_init("kaiming"))

    def weights_init(self, init_type='gaussian'):

        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun

    def forward(self, images, just_style_code=False, just_content_code=False):
        if just_style_code:
            style_code = self.encode(images, just_style_code=True)
            if self.hparams.proj_head:
                style_code = self.proj_head(style_code)
            return style_code
        if just_content_code:
            content_code = self.encode(images, just_content_code=True)
            return content_code

        content, style_code = self.encode(images)
        images_recon = self.decode(content, style_code)
        if self.hparams.proj_head:
            style_code = self.proj_head(style_code)

        if self.hparams.display_size==1: ### UNET with fully-conv input image
            if images_recon.shape[2] < images.shape[2]:
                images = images.narrow(2, 0, images_recon.shape[2])
            else:
                images_recon = images_recon.narrow(2, 0, images.shape[2])
            if images_recon.shape[3] < images.shape[3]:
                images = images.narrow(3, 0, images_recon.shape[3])
            else:
                images_recon = images_recon.narrow(3, 0, images.shape[3])

        loss = torch.mean(torch.abs(images_recon - images))
        return images_recon, style_code, content, loss

    def encode(self, images, just_style_code=False, just_content_code=False):
        style_code = self.enc_style(images)

        if self.hparams.use_style_mlp:
            style_code = self.style_mlp(style_code)


        if just_style_code:
            return style_code
        content = self.enc_content(images)
        if just_content_code:
            return content
        return content, style_code

    def decode(self, content, style):
        self.dec.apply_adain(style)
        images = self.dec(content)
        return images

    # Inference function
    def get_embedding(self, x):
        pass


def ALADIN_VGG():
    hparams = initParams(None, run_name="avgg")
    hparams.use_style_mlp = False
    hparams.latent_mode = "adain"
    hparams.size = "small"
    hparams.dim = 64
    hparams.display_size = 10
    hparams.proj_head = True
    hparams.style_dim = 896
    hparams.backbone = "avgg16"
    hparams.styleGANmod = False

    hparams.encoder_layers = None
    hparams.embedding_net = AVGG(hparams)
    hparams.embedding_net.enc_style = VGG_AdaIN(hparams)

    aladin_model = hparams.embedding_net
    return aladin_model


class ResNet50_reduced(nn.Module):
    def __init__(self, hparams):
        super(ResNet50_reduced,self).__init__()

        resnet50 = models.resnet50(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        self.fc0 = nn.Linear(in_features=2048, out_features=512, bias=True)
        self.fc1 = nn.Linear(in_features=512, out_features=128, bias=True)

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        return x
        x = nn.ReLU()(self.fc0(x))
        return self.fc1(x)


def load_model():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    checkpoint = torch.load("cogs/modules/style/aladin_vgg.pt", map_location="cpu")
    model = ALADIN_VGG()
    model.load_state_dict(checkpoint)
    model.eval()
    print('loaded AVGG')

    resnet_model = torch.load("cogs/modules/style/resnet_model.pt", map_location='cpu')
    resnet_model = resnet_model.module
    resnet_model.eval()
    print('loaded resnet')
    return model, resnet_model
