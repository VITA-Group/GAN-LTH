import torch.nn as nn
import torch

## quantizatin conv layers

_NBITS = 8
_ACTMAX = 4.0

class LinQuantSteOp(torch.autograd.Function):
    """
    Straight-through estimator operator for linear quantization
    """

    @staticmethod
    def forward(ctx, input, signed, nbits, max_val):
        """
        In the forward pass we apply the quantizer
        """
        assert max_val > 0
        if signed:
            int_max = 2 ** (nbits - 1) - 1
        else:
            int_max = 2 ** nbits
        scale = max_val / int_max
        assert scale > 0
        # return input.div(scale).round_().mul_(scale)
        return input.div(max_val).mul_(int_max).round_().mul_(scale)


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output, None, None, None

quantize = LinQuantSteOp.apply


class Conv2dQuant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.nbits = _NBITS
        self.input_signed = False
        self.input_quant = True
        self.input_max = _ACTMAX

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.input_quant:
            max_val = self.input_max
            if self.input_signed:
                min_val = -max_val
            else:
                min_val = 0.0
            input = quantize(input.clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)

        max_val = self.weight.abs().max().item()
        weight = quantize(self.weight, True, self.nbits, max_val)
        return self.conv2d_forward(input, weight)


class LinearQuant(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearQuant, self).__init__(in_features, out_features, bias)
        self.nbits = _NBITS
        self.input_signed = False
        self.input_quant = True
        self.input_max = _ACTMAX

    def linear_forward(self, input, weight):
        # self.bias is from base class nn.linear
        # weight is quantized self.weight, which is from base class nn.linear
        return F.linear(input, weight, self.bias) 
        
    def forward(self, input):
        if self.input_quant:
            max_val = self.input_max
            if self.input_signed:
                min_val = -max_val
            else:
                min_val = 0.0
            input = quantize(input.clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)

        max_val = self.weight.abs().max().item()
        weight = quantize(self.weight, True, self.nbits, max_val)
        return self.linear_forward(input, weight) 

class ChannelSelectionLayer(nn.Module):
    ''' Select some channels
    '''
    def __init__(self, selected_channel_idx):
        '''
        Args:
            selected_channel_idx: list of ints
        '''
        super(ChannelSelectionLayer, self).__init__()
        self.selected_channel_idx = selected_channel_idx
    def __call__(self, x):
        '''
        Args:
            x: Tensor, size=(N,C,W,H)
        '''
        if self.selected_channel_idx is not None:
            r = x[:,self.selected_channel_idx,:,:]
        else:
            r = x
        return r

class GenBlock(nn.Module):
    def __init__(self, conv_class, in_channels, hidden_channels, out_channels, selected_input_channel_idx=None, 
                ksize=3, pad=1, activation=nn.ReLU()):
        super(GenBlock, self).__init__()
        self.activation = activation
        self.csl = ChannelSelectionLayer(selected_channel_idx=selected_input_channel_idx)
        if selected_input_channel_idx is None:
            filtered_in_channels = in_channels 
        else:
            filtered_in_channels = len(selected_input_channel_idx)
        self.c1 = conv_class(filtered_in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = conv_class(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        self.c_sc = conv_class(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.csl(h)
        h = self.upsample_conv(h, self.c1)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        x = self.upsample_conv(x, self.c_sc)
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, bottom_width, gf_dim, latent_dim, 
                hidden_dim_lst=None, selected_input_channel_idx_lst=None, quant=False, 
                activation=nn.ReLU()):
        super(Generator, self).__init__()

        if quant:
            print('!!! Quantized model !!!')

        if quant:
            conv_class = Conv2dQuant
            linear_class = LinearQuant
        else:
            conv_class = nn.Conv2d
            linear_class = nn.Linear

        self.bottom_width = bottom_width
        self.activation = activation
        self.ch = gf_dim
        if hidden_dim_lst is None:
            hidden_dim_lst = [self.ch, self.ch, self.ch]
        if selected_input_channel_idx_lst is None:
            selected_input_channel_idx_lst = [None, None, None, None]
        
        self.l1 = linear_class(latent_dim, (self.bottom_width ** 2) * self.ch)
        # do not quantize the input noise vector
        if linear_class is LinearQuant:
            self.l1.input_quant = False

        self.block2 = GenBlock(
            conv_class=conv_class, 
            in_channels=self.ch, hidden_channels=hidden_dim_lst[0], out_channels=self.ch, 
            selected_input_channel_idx=selected_input_channel_idx_lst[0], activation=activation)
        
        self.block3 = GenBlock(
            conv_class=conv_class, 
            in_channels=self.ch, hidden_channels=hidden_dim_lst[1], out_channels=self.ch, 
            selected_input_channel_idx=selected_input_channel_idx_lst[1], activation=activation)
        
        self.block4 = GenBlock(
            conv_class=conv_class, 
            in_channels=self.ch, hidden_channels=hidden_dim_lst[2], out_channels=self.ch, 
            selected_input_channel_idx=selected_input_channel_idx_lst[2], activation=activation)
        
        self.b5 = nn.BatchNorm2d(self.ch)
        self.csl5 = ChannelSelectionLayer(selected_channel_idx=selected_input_channel_idx_lst[3])
        if selected_input_channel_idx_lst[3] is None:
            c5_in_channels = self.ch
        else:
            c5_in_channels = len(selected_input_channel_idx_lst[3])
        self.c5 = conv_class(c5_in_channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.csl5(h)
        h = nn.Tanh()(self.c5(h))
        return h


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
