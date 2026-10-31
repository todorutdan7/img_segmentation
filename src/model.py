import torch
import torch.nn as nn
import torch.nn.functional as F

# chain of 6 operations
# input tensor -> 2x(Conv2d BatchNorm2d, ReLU) -> output tensor
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=True):
        super().__init__()
        layers = []
        # first convolution in_ch -> out_ch, eg converts 1 channel grayscale slice into 32 feature maps
        # second convolution out_ch input -> out_ch output
        for a, b in [(in_ch, out_ch), (out_ch, out_ch)]:
            layers.append(nn.Conv2d(a, b, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(b))
            layers.append(nn.ReLU(inplace=True))
        # stack of layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # pass input x through the stack and return result
        return self.block(x)

# downsampling block in the encoder
# its job is to halve the spatial size while doubling the nr of feature channels
class Down(nn.Module):

    def __init__(self, in_ch, out_ch, batch_norm=True):
        super().__init__()

        # maxpool, doubleconv stack
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, batch_norm))

    def forward(self, x):
        return self.block(x)

# upsampling block in the decoder
# its job is to double the spatial size, combine it with the features from the encoder (the skip connection)
# and refine the result using a doubleconv block

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=False, batch_norm=True):
        super().__init__()
        # 2 ways to upsample controlled by bilinear flag
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_out = in_ch
        else:
            # learnable upsampling layer (deconvolution), it doubles the height and width while cutting channel count in half
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out = in_ch // 2
        self.conv = DoubleConv(up_out + skip_ch, out_ch, batch_norm)


    def forward(self, x, skip):
        # pad x with 0 so its dimensions match skip
        # cannot concat 2 tensors of different sizes

        x = self.up(x)

        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        # skip contains high resolution features like fine edges, boundaries saved from the encoder
        # x has deep, abstract features, coming up from the decoder
        # it stacks them together, if skip has 64 channels and x has 64 channels, the concat tensor has 128 channels
        return self.conv(torch.cat([skip, x], dim=1))

# in standard unet, skip connections copy everything from encoder to decoder, including background noise (ribs,spine etc)
# before allowing the skip connection features (x) into the decoder, it uses contextual information (g) to highlight
# target regions (liver) and wipe out background noise

# takes as input 2 tensors
# x the skip signal coming from the encoder, contains high resolution spatial details (edges)
# but also contains a lot of noisy background

# g - gating signal comes from a deeper layer in the decoder, contains coarse, global context (where is the tumor located?)
# but has lower spatial resolution

# g and x have diff number of channels


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 1x1 convolutions project both g and x into an intermediate channel so they can be added together
        g1 = self.wg(g)
        x1 = self.wx(x)
        # g comes from deeper layer, its height x width is smaller than x -> resize g1 to match higher dimensions of x1
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        # add gating signal and skip signal together, where both agree a region is important, the activation gets stronger
        # relu wipes out negative activations
        # psi uses a 1v1 conv to collapse the channels down to a single channel, followed by sigmoid activation
        # alpha is a 2d heatmap with pixel vals between 0 and 1
        # if alpha = close to 1 tumor/liver keep it
        # if alpha = close to 0 background noise
        alpha = self.psi(self.relu(g1 + x1))
        # multiply original skip feature map x element by element with the attention weights alpha
        return x * alpha


def init_conv_bn(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_filters=32, depth=4, batch_norm=True, dropout=0.0, bilinear=False, **_):
        super().__init__()
        # depth 4, after we overflow
        self.depth = depth
        # f holds [32,64,128,256,512], multiply by 2 the nr of filters are each level
        f = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.filters = f
        # the encoder
        # incoming layer
        # doubleconv takes the raw grayscale image (1 channel) and outputs 32 channels (f[0])
        self.inc = DoubleConv(in_channels, f[0], batch_norm)
        # loop 4 times to build the downsampling blocks
        # down 1 32 -> 64, 64 -> 128, ... 256 -> 512, 512 is the bottleneck at the bottom of the unet
        self.downs = nn.ModuleList(
            [Down(f[i], f[i + 1], batch_norm) for i in range(depth)])

        # if dropout: zero some channels at deepest aprt of the network to prevent overfitting
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # decoder and attention gates
        # built using reversed(range(depth)) which gives [3,2,1,0]
        # because the decoder starts at the bottom (512 channels)
        # and moves UP to the top (32 channels)

        # takes the deeper decoders signal (g_ch = 512), the skip connection(x_ch = 256)
        # and compresses them into an intermediate channel size (inter_ch = 128)
        self.gates = nn.ModuleList(
            [AttentionGate(g_ch=f[i + 1], x_ch=f[i], inter_ch=max(f[i] // 2, 1))
             for i in reversed(range(depth))])
        # takes the deeper signal (512), the filtered skip connection (256) and outputs the next level (256)
        self.ups = nn.ModuleList(
            [Up(f[i + 1], f[i], f[i], bilinear, batch_norm)
             for i in reversed(range(depth))])


        # output layer: the network finishes at f[0] (32 channels)
        # a final 1x1 convolution compresses those 32 channels down to num_classes (3 classes)
        # background, liver, tumor

        self.head = nn.Conv2d(f[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        init_conv_bn(self)

    def forward(self, x):
        # journey of image tensor x

        # first layer
        # skips is a stack to save high resolution features so we use them later
        skips = []
        # pass the raw 1 channel image into the first doubleconv to get 32 channels
        x = self.inc(x)
        # saves a copy of these 32 channel features into the list
        skips.append(x)

        # encoder
        # tensor x loops through the 4 down blocks
        for down in self.downs:
            # at each step, x gets shrunk by half but deeper (channels: 64,128,256,512)
            x = down(x)
            # after each single step down, a copy of x is saved into skips stack
            skips.append(x)
        # the last item added to the skips list was the output of the final down block
        # which is the deepest 512 channel tensor
        # don't want to use this as a skip connection (this is the actual bottleneck feature map)
        # pop() removes it from the list and sets it as our current tensor x
        # self.drop -> it passes this bottleneck tensor though the dropout layer to prevent overfit
        x = self.drop(skips.pop())

        # the decoder
        # travel back up to the U
        # iterate through the attentiongates and up blocks at the same time

        #
        for gate, up in zip(self.gates, self.ups):
            # grab the most recently saved skip connection
            skip = skips.pop()
            # gate(x, skip) -> passes the deep features (x) and the saved skip features into the gate
            # then the gate outputs a filtered version of skip where background noise is erased
            # up(x, gate(x,skip)) -> passes the deep features (x) and the filtered skip connection into the up nlock
            # the up block doubles the size of x, concatenates it with the filtered skip and combines them together using doubleconv
            # loop repeats 4 times, until the img is back to original 256x256 resolution
            x = up(x, gate(x, skip))

        # after loop is done, x is back to 32x256x256
        # a 1x1 conv squashes those 32 channels down into exactly 3 channels
        # probability of background
        # probability of liver
        # probability of tumor
        # this output tensor is what gets compared to the mask to calculate the loss
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_filters=32, depth=4, batch_norm=True, dropout=0.0, bilinear=False, **_):

        super().__init__()
        self.depth = depth
        f = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.filters = f

        self.inc = DoubleConv(in_channels, f[0], batch_norm)

        self.downs = nn.ModuleList(
            [Down(f[i], f[i + 1], batch_norm) for i in range(depth)])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()


        # no self.gates - that is the entire difference
        self.ups = nn.ModuleList(
            [Up(f[i + 1], f[i], f[i], bilinear, batch_norm)
             for i in reversed(range(depth))])

        self.head = nn.Conv2d(f[0], num_classes, kernel_size=1)
        init_conv_bn(self)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.drop(skips.pop())
        for up in self.ups:
            # the skip arrives unfiltered compared with AttentionUNet.forward
            # which wraps this same tensor in gate(x, skip) first
            x = up(x, skips.pop())

        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# U-Net, Attention U-Net and U-Net++ with an ImageNet-pretrained ResNet-50 encoder
# ---------------------------------------------------------------------------
# The encoder half is a ResNet-50 that already learnt edges, textures and
# shapes on 1.2 million ImageNet photos, so only the decoder starts from
# scratch. ImageNet images have 3 channels, which a 2.5D input (the slice and
# its two neighbours) fills naturally.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet50Encoder(nn.Module):
    """ResNet-50 without its classifier, returning the feature maps at 1/2,
    1/4, 1/8, 1/16 (the skips) and 1/32 (the bottleneck) of the input size"""
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        # the ImageNet weights are downloaded once (~100 MB) to torch's cache
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if in_channels != 3:
            # each input channel gets the mean of the pretrained RGB filters,
            # scaled so the first layer's response keeps its size
            w = net.conv1.weight.data
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            net.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_channels, 1, 1) * 3 / in_channels
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        # the data arrives normalised to -1..1; ImageNet weights expect their
        # own per-channel mean and std, applied here inside the model
        mean = IMAGENET_MEAN if in_channels == 3 else (sum(IMAGENET_MEAN) / 3,) * in_channels
        std = IMAGENET_STD if in_channels == 3 else (sum(IMAGENET_STD) / 3,) * in_channels
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        x = ((x * 0.5 + 0.5) - self.mean) / self.std
        feats = [self.stem(x)]                  # 1/2, 64 channels
        x = self.pool(feats[0])
        for layer in self.layers:               # 1/4 256, 1/8 512, 1/16 1024, 1/32 2048
            x = layer(x)
            feats.append(x)
        return feats


class UpBlock(nn.Module):
    """decoder step: double the size, join the skip - filtered by an attention
    gate in the attention variant - and refine with a DoubleConv"""

    def __init__(self, in_ch, skip_ch, out_ch, attention=False):
        super().__init__()
        self.gate = AttentionGate(in_ch, skip_ch, max(skip_ch // 2, 1)) if attention and skip_ch else None
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        if skip is not None and self.gate is not None:
            skip = self.gate(x, skip)           # x, not yet upsampled, is the gating signal
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net (attention=False) or Attention U-Net (attention=True) whose encoder
    is a ResNet-50. The decoder climbs 1/32 -> 1/2 joining the four skips, then
    a last step without skip brings it back to full size."""
    decoder_channels = (256, 128, 64, 32, 16)

    def __init__(self, in_channels=3, num_classes=3, attention=False, pretrained=True,
                 dropout=0.0, **_):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels, pretrained)
        enc = ResNet50Encoder.channels
        ins = (enc[4],) + self.decoder_channels[:-1]        # 2048, 256, 128, 64, 32
        skips = (enc[3], enc[2], enc[1], enc[0], 0)        # 1024, 512, 256, 64, none
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([UpBlock(i, s, o, attention)
                                     for i, s, o in zip(ins, skips, self.decoder_channels)])
        self.head = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
        # decoder and head start fresh; the encoder keeps its ImageNet weights
        init_conv_bn(self.blocks)
        init_conv_bn(self.head)

    @property
    def gates(self):
        # read by attention_maps(); None for the plain U-Net variant
        g = [b.gate for b in self.blocks if b.gate is not None]
        return g or None

    def forward(self, x):
        f = self.encoder(x)
        x = self.drop(f[4])
        for block, skip in zip(self.blocks, (f[3], f[2], f[1], f[0], None)):
            x = block(x, skip)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetUNetPlusPlus(nn.Module):
    """U-Net++ (Zhou et al., 2018) with the same ResNet-50 encoder.

    A U-Net joins each encoder level straight to the decoder. U-Net++ puts a
    row of nodes in between: node (i, j) on level i takes everything already
    on that level, (i, 0) to (i, j-1), plus node (i+1, j-1) from one level
    down, upsampled. The encoder features are so refined step by step before
    they meet the decoder. The last node of each row, i + j = 4, is the
    decoder path proper and has the U-Net's decoder widths; the nodes before
    it keep the width of their encoder level (the layout of
    segmentation_models_pytorch). As in the U-Net, a last step without skip
    brings the 1/2 size back to full size.
    """
    decoder_channels = ResNetUNet.decoder_channels

    def __init__(self, in_channels=3, num_classes=3, pretrained=True, dropout=0.0, **_):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels, pretrained)
        enc = ResNet50Encoder.channels
        n = len(enc) - 1                                    # 4 levels with skips

        def width(i, j):
            return self.decoder_channels[n - 1 - i] if i + j == n else enc[i]

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.nodes = nn.ModuleDict()
        for j in range(1, n + 1):
            for i in range(n + 1 - j):
                below = enc[i + 1] if j == 1 else width(i + 1, j - 1)
                # the j nodes already on level i, each enc[i] wide, are its skip
                self.nodes[f"{i}_{j}"] = UpBlock(below, enc[i] * j, width(i, j))
        self.last = UpBlock(self.decoder_channels[-2], 0, self.decoder_channels[-1])
        self.head = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
        init_conv_bn(self.nodes)
        init_conv_bn(self.last)
        init_conv_bn(self.head)

    def forward(self, x):
        f = self.encoder(x)
        f[4] = self.drop(f[4])
        rows = [[feat] for feat in f]                       # rows[i][j] is node (i, j)
        n = len(f) - 1
        # column by column: every node of column j needs column j-1 done
        for j in range(1, n + 1):
            for i in range(n + 1 - j):
                node = self.nodes[f"{i}_{j}"]
                rows[i].append(node(rows[i + 1][j - 1], torch.cat(rows[i], dim=1)))
        return self.head(self.last(rows[0][n]))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


ARCHES = {"attention_unet": AttentionUNet, "unet": UNet,
          "attention_unet_r50": lambda **kw: ResNetUNet(attention=True, **kw),
          "unet_r50": lambda **kw: ResNetUNet(attention=False, **kw),
          "unetpp_r50": ResNetUNetPlusPlus}


@torch.no_grad()
def attention_maps(model, x):
    # the alpha map each gate produced
    gates = getattr(model, "gates", None)
    if gates is None:
        raise ValueError(
            f"{type(model).__name__} has no attention gates - there are no alpha "
            f"maps to read. This is expected for arch='unet'.")
    maps, handles = [], []
    for gate in gates:
        handles.append(gate.psi.register_forward_hook(
            lambda _m, _i, out: maps.append(out.detach().cpu())))
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
    return maps



def load_checkpoint(ckpt_path, device=None):

    from dataclasses import fields
    from src.config import Config, get_device
    device = device or get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    names = {f.name for f in fields(Config)}
    saved = dict(ck["config"])
    # checkpoints from before the nifti work do not record a format, and the
    # default is now nifti - but every one of them was trained on the pngs
    saved.setdefault("data_format", "png")
    cfg = Config(**{k: v for k, v in saved.items() if k in names})
    # no ImageNet download: the checkpoint's own weights replace them right away
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck



def build_model(cfg, pretrained=None):
    """the network cfg describes. pretrained (ResNet-50 models only) overrides
    cfg.encoder_weights; load_checkpoint passes False to skip the download"""
    arch = getattr(cfg, "arch", "attention_unet")
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHES)}")
    kwargs = dict(
        in_channels=getattr(cfg, "in_slices", 1),
        num_classes=cfg.num_classes,
        base_filters=cfg.base_filters,
        depth=cfg.depth,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        bilinear=cfg.bilinear,
    )
    if arch.endswith("_r50"):
        kwargs["pretrained"] = (cfg.encoder_weights == "imagenet") if pretrained is None else pretrained
    return ARCHES[arch](**kwargs)
