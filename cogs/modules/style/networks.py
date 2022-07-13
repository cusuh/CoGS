from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class UNet(nn.Module):
    def __init__ (self, hparams):
        super(UNet, self).__init__()

        self.hparams = hparams

        # Parameters
        input_channels = 3
        # activ = "relu"
        activ = "lrelu"
        pad_type = "reflect"
        n_res = 4 # number of residual blocks in content encoder/decoder
        mlp_dim = 256 # number of filters in MLP

        if self.hparams.use_style_mlp:
            self.style_mlp = nn.Linear(self.hparams.style_dim, self.hparams.style_dim)

        # Style encoder
        self.enc_style = hparams.enc_style(self.hparams) if hparams.enc_style is not None else StyleEncoder(4, input_channels, hparams.dim, hparams.style_dim, norm='none', activ=activ, pad_type=pad_type)
        # Content encoder
        if self.hparams.size == "medium" or self.hparams.size == "big":
            self.enc_content = ContentEncoder(2 if self.hparams.size == "medium" else 3, n_res, input_channels, hparams.dim, 'in', activ, pad_type=pad_type)
        else:
            self.enc_content = UNet_ContentEncoder(self.enc_style, n_res, input_channels, hparams.dim, 'in', activ, pad_type=pad_type)
        # Image decoder
        self.dec = UNet_Decoder(self.enc_style, n_res, self.enc_content.output_dim, input_channels, res_norm='adain', activ=activ, pad_type=pad_type)

        # Network weight initialization
        self.apply(self.weights_init("kaiming"))

    def weights_init (self, init_type='gaussian'):

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
            return style_code
        if just_content_code:
            content_code = self.encode(images, just_content_code=True)
            return content_code

        content, style_code = self.encode(images)
        images_recon = self.decode(content, style_code)

        if self.hparams.display_size==1: ### UNET with fully-conv input image
            if images_recon.shape[2] < images.shape[2]:
                images = images.narrow(2, 0, images_recon.shape[2])
            else:
                images_recon = images_recon.narrow(2, 0, images.shape[2])
            if images_recon.shape[3] < images.shape[3]:
                images = images.narrow(3, 0, images_recon.shape[3])
            else:
                images_recon = images_recon.narrow(3, 0, images.shape[3])

        # loss = torch.mean(torch.abs(images_recon - images))
        loss = None
        return images_recon, style_code, loss

    def encode(self, images, just_style_code=False, just_content_code=False):
        style_code = self.enc_style(images)
        # style_code = torch.div(style_code, style_code.pow(2).sum(1, keepdim=True).sqrt())

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


class UNet_ContentEncoder(nn.Module):
    def __init__(self, enc_style, n_res, input_dim, dim, norm, activ, pad_type):
        super(UNet_ContentEncoder, self).__init__()
        self.model = []

        params_per_layer = []
        self.filter_per_layer = []
        encoder_layers = [l for l in enc_style.encoder_layers if l[0]=="c"]
        for l in range(len(encoder_layers)):
            [l_type, layer] = encoder_layers[l]
            prev = input_dim if l==0 else encoder_layers[l-1][1].weight.shape[0]
            params_per_layer.append((prev, layer.weight.shape[0]))
            self.filter_per_layer.append(layer.weight.shape[0])

        [prev, dim] = params_per_layer[0]
        self.model += [Conv2dBlock(prev, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for ppl in range(1, len(params_per_layer)):
            [prev, dim] = params_per_layer[ppl]
            self.model += [Conv2dBlock(prev, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        dim = params_per_layer[-1][1]

        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class UNet_Decoder(nn.Module):
    def __init__(self, enc_style, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(UNet_Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, "in", activ, pad_type=pad_type)]
        self.model += [AdaptiveInstanceNorm2d(dim)]
        # upsampling blocks
        params_per_layer = []
        self.filter_per_layer = []
        encoder_layers = [l for l in enc_style.encoder_layers if l[0]=="c"]
        for l in range(len(encoder_layers)):
            [l_type, layer] = encoder_layers[l]
            prev = output_dim if l==0 else encoder_layers[l-1][1].weight.shape[0]
            params_per_layer.append((prev, layer.weight.shape[0]))
            self.filter_per_layer.append(layer.weight.shape[0])

        params_per_layer.reverse()

        for ppl in range(len(params_per_layer)-1):
            [prev, dim] = params_per_layer[ppl]

            new_layers = []
            if prev!=dim:
                new_layers.append(nn.Upsample(scale_factor=2))

            new_layers.append(Conv2dBlock(dim, prev, 5, 1, 2, norm='adain', activation=activ, pad_type=pad_type))
            self.model += new_layers

        dim = params_per_layer[-1][1]

        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)


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


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

# ======= MUNIT classes start

class EmbeddingNet_MUNIT(nn.Module):
    def __init__(self, hparams):
        super(EmbeddingNet_MUNIT, self).__init__()

        self.hparams = hparams

        # Parameters
        input_channels = 3
        # activ = "relu"
        activ = "lrelu"
        pad_type = "reflect"
        n_downsample = 2 # number of downsampling layers in content encoder
        n_res = 4 # number of residual blocks in content encoder/decoder
        mlp_dim = 256 # number of filters in MLP

        beta1 = 0.5
        beta2 = 0.999
        weight_decay = 0.0001

        if self.hparams.use_style_mlp:
            self.style_mlp = nn.Linear(self.hparams.style_dim, self.hparams.style_dim)


        # Style encoder
        # self.enc_style = hparams.enc_style if hparams.enc_style is not None else StyleEncoder(4, input_channels, hparams.dim, hparams.style_dim, norm='none', activ=activ, pad_type=pad_type)
        self.enc_style = hparams.enc_style(self.hparams) if hparams.enc_style is not None else StyleEncoder(4, input_channels, hparams.dim, hparams.style_dim, norm='none', activ=activ, pad_type=pad_type)
        # Content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_channels, hparams.dim, 'in', activ, pad_type=pad_type)
        # Image decoder
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_channels, res_norm='adain', activ=activ, pad_type=pad_type)
        # MLP to generate AdaIN parameters
        self.mlp = MLP(hparams.style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)


        # params = list(self.enc_style.parameters())+list(self.enc_content.parameters())+list(self.dec.parameters())+list(self.mlp.parameters())
        params = list(self.parameters())

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

    def update_learning_rate(self):
        self.scheduler.step()

    def forward(self, images, just_style_code=False, just_content_code=False):
        if just_style_code:
            style_code = self.encode(images, just_style_code=True)
            return style_code

        if just_content_code:
            content_code = self.encode(images, just_content_code=True)
            return content_code

        content, style_code = self.encode(images)
        images_recon = self.decode(content, style_code)

        loss = torch.mean(torch.abs(images_recon - images))
        return images_recon, style_code, loss

    def encode(self, images, just_style_code=False, just_content_code=False):
        style_code = self.enc_style(images)
        style_code = torch.div(style_code, style_code.pow(2).sum(1, keepdim=True).sqrt())

        if self.hparams.use_style_mlp:
            style_code = self.style_mlp(style_code.view((style_code.shape[0], style_code.shape[1])))

        if just_style_code:
            return style_code

        if self.hparams.pre_training:
            content = self.enc_content(images)
        else:
            content = self.enc_content(images)

        if just_content_code:
            return content
        return content, style_code

    def decode (self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    # Inference function
    def get_embedding(self, x):
        pass

# ======= MUNIT classes end

class ResNet50_bam(nn.Module):
    def __init__(self):
        super(ResNet50_bam,self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        self.fc0 = nn.Linear(in_features=2048, out_features=20, bias=True)

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        return x,self.fc0(x)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        return x


class ResNet50_reduced(nn.Module):
    def __init__(self):
        super(ResNet50_reduced,self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        self.fc0 = nn.Linear(in_features=2048, out_features=512, bias=True)
        self.fc1 = nn.Linear(in_features=512, out_features=128, bias=True)

    def forward(self,x):
        x = self.base(x)
        return x
        x = x.view(x.size(0),-1)
        x = nn.ReLU()(self.fc0(x))
        return self.fc1(x)


class ResNet50_reduced_2(nn.Module):
    def __init__(self):
        super(ResNet50_reduced_2,self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        self.fc0 = nn.Linear(in_features=2048, out_features=512, bias=True)
        # self.fc1 = nn.Linear(in_features=512, out_features=128, bias=True)

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        # x = nn.ReLU()(self.fc0(x))
        return self.fc0(x)


class ResNet50_imagenet_stock(nn.Module):
    def __init__(self,res,out):
        super(ResNet50_imagenet_stock,self).__init__()

        if res=='50':
            resnet50 = torchvision.models.resnet50(pretrained=False)
        elif res=='152':
            print('resnet 152')
            resnet50 = torchvision.models.resnet152(pretrained=False)
        else:
            raise Exception('res should be either "50" or "152"')

        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        self.fc0 = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.fc1 = nn.Linear(in_features=2048, out_features=out, bias=True)

    def forward(self,x):
        x = self.base(x)
        f = x.view(x.size(0),-1)
        return [self.fc0(f),self.fc1(f)]


class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='in', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]

                std[std <= 0] =  .001
                std[std > 10000] =  10000

                mean[mean < -10000] =  -10000
                mean[mean >  10000] =   10000

                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################
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
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class AdaIN (torch.nn.Module):

    def __init__ (self, hparams):
        super(AdaIN, self).__init__()

        self.hparams = hparams
        self.device = torch.device((f'cuda:{self.hparams.gpu_index}' if self.hparams.gpu_index is not None else "cuda") if self.hparams.use_gpu else "cpu")
        self.gauss_noise = {}

    def forward(self, f_vol):
        batch_size = f_vol.shape[0]
        adain_batch = []

        for bs in range(batch_size):
            f_map = f_vol[bs]

            noise_shape = f'{f_map.shape[0]}_{f_map.shape[1]}_{f_map.shape[2]}'
            if noise_shape not in self.gauss_noise.keys():
                self.gauss_noise[noise_shape] = torch.randn(f_map.shape).to(self.device)

            means = []
            stds = []
            no_channels = f_map.shape[0]

            for c in range(no_channels):
                means.append(torch.mean(f_map[c]))
                stds.append(torch.std(f_map[c]))

            adain = means + stds
            adain = torch.cat([v.view(1) for v in adain])

            adain_batch.append(adain)

        adain_batch = torch.cat(adain_batch, dim=-1)
        adain_batch = adain_batch.expand(1, adain_batch.shape[0])
        adain_batch = adain_batch.view(batch_size, -1)
        return adain_batch


class VLAE_Encoder (nn.Module):
    def __init__(self, hparams, encoder_layers=None):
        super(VLAE_Encoder, self).__init__()

        print("")
        self.hparams = hparams

        self.model = []
        self.encoder_layers = encoder_layers if encoder_layers else self.hparams.encoder_layers
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
                if self.hparams.latent_mode == "fc":
                    num_params, last_dim = self.get_fc_size(layer, last_channels, last_dim)
                    last_channels = layer.weight.shape[0]
                    self.model.append((f'linear_{conv_count}', nn.Linear(num_params, int(self.hparams.fc_component_latent_size)) ))

                elif self.hparams.latent_mode == "adain":
                    self.model.append((f'adain_{conv_count}', AdaIN(hparams) ))
                conv_count += 1

            elif l_type=="p":
                self.model.append((f'pool_{pool_count}', layer))
                last_dim = int(last_dim/layer.stride)
                pool_count += 1

        self.model.append(("flatten", Flatten()))
        self.model = nn.Sequential(OrderedDict(self.model))

        self.layers = dict(self.model.named_children())
        # self.relu = torch.nn.functional.relu
        self.relu = torch.nn.functional.leaky_relu

    def forward(self, data):
        self.layers = dict(self.model.named_children())
        batch_count = data.shape[0]

        latent = []
        cat_dim = 1

        conv_count = 0
        pool_count = 0
        for l in range(self.num_layers):

            l_type = self.encoder_layers[l][0]

            if l_type == "c":

                feat = self.relu( self.layers[f'conv_{conv_count}'](data) )
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

        del data
        latent = torch.cat(latent, dim=cat_dim)
        return latent

    def get_feat_map_adain (self, f_map):

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
    def get_fc_size (self, conv, channels, dim):

        s = conv.stride[0]
        p = conv.padding[0]
        k = conv.kernel_size[0]
        i = dim

        out_size = (i-k+2*p)/s + 1
        return conv(torch.rand(1, channels, dim, dim)).data.view(1, -1).size(1), int(out_size)


def get_encoder_layers(args):
    vanilla_types = ["small", "medium", "big"]
    vanilla_encoder_layers = [[] for t in vanilla_types]

    unet_types = ["thin", "wide", "small", "medium", "big"]
    unet_encoder_layers = [[] for t in unet_types]

    # Small
    vanilla_encoder_layers[0].append(("c", nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)))
    vanilla_encoder_layers[0].append(("c", nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)))

    # Medium - First few VGG-16 layers
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(3, 64, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(64, 64, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(64, 128, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(128, 128, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(128, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(256, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[1].append(("c", nn.Conv2d(256, 256, kernel_size=3, padding=1)))

    # Large - VGG-19 layers
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(3, 64, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(64, 64, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(64, 128, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(128, 128, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(128, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(256, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(256, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(256, 256, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(256, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("p", nn.MaxPool2d(kernel_size=2, stride=2)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))
    vanilla_encoder_layers[2].append(("c", nn.Conv2d(512, 512, kernel_size=3, padding=1)))

    # U-Net Thin
    unet_encoder_layers[0].append(("c", nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)))
    unet_encoder_layers[0].append(("c", nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[0].append(("c", nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)))

    # U-Net Wide
    unet_encoder_layers[1].append(("c", nn.Conv2d(3, 128, kernel_size=7, stride=1, padding=3)))
    unet_encoder_layers[1].append(("c", nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[1].append(("c", nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)))

    # U-Net Small
    unet_encoder_layers[2].append(("c", nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)))
    unet_encoder_layers[2].append(("c", nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[2].append(("c", nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)))

    # U-Net Medium
    unet_encoder_layers[3].append(("c", nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)))
    unet_encoder_layers[3].append(("c", nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[3].append(("c", nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[3].append(("c", nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[3].append(("c", nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[3].append(("c", nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)))

    # U-Net Large
    unet_encoder_layers[4].append(("c", nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)))
    unet_encoder_layers[4].append(("c", nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)))

    configs = {}
    configs["munit"] = [vanilla_types, vanilla_encoder_layers]
    configs["vlae_fc"] = configs["munit"]
    configs["vlae_adain"] = configs["munit"]
    configs["unet"] = [unet_types, unet_encoder_layers]

    return configs[args.backbone][1][configs[args.backbone][0].index(args.size)]
    