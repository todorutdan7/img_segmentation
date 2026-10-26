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


ARCHES = {"attention_unet": AttentionUNet, "unet": UNet}


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
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck



def build_model(cfg):
    from src.config import IN_CHANNELS
    arch = getattr(cfg, "arch", "attention_unet")
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHES)}")
    return ARCHES[arch](
        in_channels=IN_CHANNELS,
        num_classes=cfg.num_classes,
        base_filters=cfg.base_filters,
        depth=cfg.depth,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        bilinear=cfg.bilinear,
    )
