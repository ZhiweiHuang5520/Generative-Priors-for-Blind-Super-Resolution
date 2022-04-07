import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)
        y = self.gamma * out + input
 
        return self.gamma * out + input

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # upsample.shape = torch.Size([8, 3, 32, 32])
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=False)

        # block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.atrous_block1(x)

        # block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.atrous_block6(x)

        # block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.atrous_block12(x)

        # block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.atrous_block18(x)

        # torch.cat.shape = torch.Size([8, 15, 32, 32])
        # conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
    
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
    


class Encoder(nn.Module):
    """
    Args:
            up_ch (int): Channel number of upsample features.
            num_feat (int): Channel number of intermediate features.
                Default: 64
            num_block (int): Block number in the trunk network. Defaults: 23
            num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, up_ch = 64, sf = 2, num_block = 23, num_feat = 64, num_grow_ch = 32):

        super().__init__()
        self.sf = sf
        # upsample
        self.conv_up1 = nn.Conv2d(3, up_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.aspp1 = ASPP(up_ch, up_ch * 2)
#         self.att1 = SelfAttention(up_ch)
        
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.aspp2 = ASPP(num_feat, num_feat * 4)
        
        self.flatten = nn.Flatten()
        self.adap1 = nn.AdaptiveAvgPool2d((4,4))
        self.linear = nn.Linear(num_feat * 16, 1000)
        
        
        
#         self.down1_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=2, padding=0, bias=False)
#         self.down1_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=False)
        
#         self.down2_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=2, padding=0, bias=False)
#         self.down2_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=False)
        
#         self.down3_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=2, padding=0, bias=False)
#         self.down3_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=False)
        
#         self.down4_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=2, padding=0, bias=False)
#         self.down4_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=False)
        

    def forward(self, im):
        # upsample
        feat1 = self.lrelu(self.conv_up1(F.interpolate(im, scale_factor=self.sf, mode='nearest')))
        up_feat =  self.aspp1(feat1) 
        
        
        inter_feat = self.conv_first(im)
        body_feat = self.conv_body(self.body(inter_feat))
        inter_feat = inter_feat + body_feat
        
        pre_latent = self.flatten(self.adap1(inter_feat))#.view(inter_feat.shape[0], inter_feat.shape[1] * 64)
        latent = self.linear(pre_latent)
    
        final_feat = self.aspp2(inter_feat)
        
        
#         down_feat1 = self.down1_conv1(inter_feat)
#         down_feat1 = self.down1_conv2(down_feat1)
#         down_feat1 = self.lrelu(down_feat1)
        
#         down_feat2 = self.down2_conv1(down_feat1)
#         down_feat2 = self.down2_conv2(down_feat2)
#         down_feat2 = self.lrelu(down_feat2)
        
#         down_feat3 = self.down3_conv1(down_feat2)
#         down_feat3 = self.down3_conv2(down_feat3)
#         down_feat3 = self.lrelu(down_feat3)
        
#         down_feat4 = self.down4_conv1(down_feat3)
#         down_feat4 = self.down4_conv2(down_feat4)
        
        
        return up_feat, inter_feat, final_feat
    
class Decoder(nn.Module):

    def __init__(self, up_ch = 64, sf= 2, num_feat = 64, num_out_ch = 3):
        super().__init__()
        
        # upsample
        
        self.Tconv_up1 = nn.ConvTranspose2d(num_feat*4, num_feat*4, sf, stride=sf)
        
        self.conv_up1 = nn.Conv2d(num_feat*4, num_feat*4, 3, 1, 1)
        
        self.conv_hr = nn.Conv2d(num_feat*4 + up_ch * 2, num_feat*4, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat*4, num_out_ch, 3, 1, 1)
        
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, up_feat, inter_feat, final_feat):
        # upsample
        
        feat = self.lrelu(self.conv_up1(self.Tconv_up1(final_feat)))
        feed = torch.cat([up_feat, feat],dim = 1)
        out = self.conv_last(self.lrelu(self.conv_hr(feed)))
        
        return out