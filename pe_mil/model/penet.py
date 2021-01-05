
import math
import torch
import torch.nn as nn
import random



## PENet blocks


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Based on the paper:
    "Squeeze-and-Excitation Networks"
    by Jie Hu, Li Shen, Gang Sun
    (https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, num_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(nn.Linear(num_channels, num_channels // reduction),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(num_channels // reduction, num_channels),
                                    nn.Sigmoid())

    def forward(self, x):
        num_channels = x.size(1)

        # Squeeze
        z = self.squeeze(x)
        z = z.view(-1, num_channels)

        # Excite
        s = self.excite(z)
        s = s.view(-1, num_channels, 1, 1, 1)

        # Apply gate
        x = x * s

        return x


class PENetBottleneck(nn.Module):
    """PENet bottleneck block, similar to a pre-activation ResNeXt bottleneck block.

    Based on the paper:
    "Aggregated Residual Transformations for Deep Nerual Networks"
    by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    (https://arxiv.org/abs/1611.05431).
    """

    expansion = 2

    def __init__(self, in_channels, channels, block_idx, total_blocks, cardinality=32, stride=1):
        super(PENetBottleneck, self).__init__()
        mid_channels = cardinality * int(channels / cardinality)
        out_channels = channels * self.expansion
        self.survival_prob = self._get_survival_prob(block_idx, total_blocks)

        self.down_sample = None
        if stride != 1 or in_channels != channels * PENetBottleneck.expansion:
            self.down_sample = nn.Sequential(
                nn.Conv3d(in_channels, channels * PENetBottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(channels * PENetBottleneck.expansion // 16, channels * PENetBottleneck.expansion))

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(out_channels // 16, out_channels)
        self.norm3.is_last_norm = True
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.se_block = SEBlock(out_channels, reduction=16)

    @staticmethod
    def _get_survival_prob(block_idx, total_blocks, p_final=0.5):
        """Get survival probability for stochastic depth. Uses linearly decreasing
        survival probability as described in "Deep Networks with Stochastic Depth".

        Args:
            block_idx: Index of residual block within entire network.
            total_blocks: Total number of residual blocks in entire network.
            p_final: Survival probability of the final layer.
        """
        return 1. - block_idx / total_blocks * (1. - p_final)

    def forward(self, x):
        x_skip = x if self.down_sample is None else self.down_sample(x)

        # Stochastic depth dropout
        # if self.training and random.random() > self.survival_prob:
        #     return x_skip

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.se_block(x)
        x += x_skip

        x = self.relu3(x)

        return x


class PENetEncoder(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, cardinality, block_idx, total_blocks, stride=1):
        super(PENetEncoder, self).__init__()

        # Get PENet blocks
        penet_blocks = [PENetBottleneck(in_channels, channels, block_idx, total_blocks, cardinality, stride)]

        for i in range(1, num_blocks):
            penet_blocks += [PENetBottleneck(channels * PENetBottleneck.expansion, channels,
                                           block_idx + i, total_blocks, cardinality)]
        self.penet_blocks = nn.Sequential(*penet_blocks)

    def forward(self, x):
        x = self.penet_blocks(x)
        return x

class PENetASPPool(nn.Module):
    """Atrous Spatial Pyramid Pooling layer.

    Based on the paper:
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    by Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
    (https://arxiv.org/abs/1706.05587).
    """
    def __init__(self, in_channels, out_channels):
        super(PENetASPPool, self).__init__()

        self.mid_channels = out_channels // 4
        self.in_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
                                     nn.GroupNorm(out_channels // 16, out_channels),
                                     nn.LeakyReLU(inplace=True))

        self.conv1 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                   nn.Conv3d(out_channels, self.mid_channels, kernel_size=1))
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.out_conv = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=1),
                                      nn.GroupNorm(out_channels // 16, out_channels),
                                      nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.in_conv(x)

        # Four parallel paths with different dilation factors
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)
        x_4 = x_4.expand(-1, -1, x_1.size(2), x_1.size(3), x_1.size(4))

        # Combine parallel pathways
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.norm(x)
        x = self.relu(x)

        x = self.out_conv(x)

        return x

class GAPLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Global average pooling (3D) followed by a linear layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels
        """
        super(GAPLinear, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, 1)
        self.fc.is_output_head = True

    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class PENetLateral(nn.Module):
    """Lateral connection layer for PENet."""
    def __init__(self, in_channels, out_channels):
        super(PENetLateral, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, x_skip):
        # Reduce number of channels in skip connection
        x_skip = self.conv(x_skip)
        x_skip = self.norm(x_skip)
        x_skip = self.relu(x_skip)

        # Add reduced feature map
        x += x_skip

        return x

class PENetDecoder(nn.Module):
    """Decoder (up-sampling layer) for PENet"""
    def __init__(self, skip_channels, in_channels, mid_channels, out_channels, kernel_size=4, stride=2):
        super(PENetDecoder, self).__init__()

        if skip_channels > 0:
            self.lateral = PENetLateral(skip_channels, in_channels)

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu3 = nn.LeakyReLU(inplace=True)

    def forward(self, x, x_skip=None):
        if x_skip is not None:
            x = self.lateral(x, x_skip)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

#################################################################################




class PENet(nn.Module):

    def __init__(self, net_input_channels=3, cardinality=32, pretrained_ckpt_fn='', device=0,
                        num_classes=1, init_method=None, do_classify=False, **kwargs):
        super(PENet, self).__init__()
        self.cardinality = cardinality
        self.do_classify = do_classify
        self.num_classes = num_classes
        self.device = device

        self.net_input_channels = net_input_channels
        self.in_channels = 64
        self.model_depth = 50

        self.in_conv = nn.Sequential(
                        nn.Conv3d(self.net_input_channels, self.in_channels, 
                                  kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                        nn.GroupNorm(self.in_channels // 16, self.in_channels),
                        nn.LeakyReLU(inplace=True))

        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Encoders, 50
        encoder_config = [3, 4, 6]
        total_blocks = sum(encoder_config)
        block_idx = 0

        self.encoders = nn.ModuleList()
        for i, num_blocks in enumerate(encoder_config):
            out_channels = 2 ** i * 128
            stride = 1 if i == 0 else 2
            encoder = PENetEncoder(self.in_channels, out_channels, num_blocks, self.cardinality,
                                  block_idx, total_blocks, stride=stride)
            self.encoders.append(encoder)
            self.in_channels = out_channels * PENetBottleneck.expansion
            block_idx += num_blocks

        self.asp_pool = PENetASPPool(1024, 256)

        if self.do_classify:
            self.classifier = GAPLinear(256, num_classes)

        # Decoders
        decoder_config = [(0, 256, 256, 128), (512, 128, 128, 64), (256, 64, 64, 64), (64, 64, 64, 64)]
        total_blocks = 2 * len(decoder_config)
        block_idx = total_blocks - 1

        self.decoders = nn.ModuleList()
        for i, (skip_channels, in_channels, mid_channels, out_channels) in enumerate(decoder_config):
            is_last_decoder = (i == len(decoder_config) - 1)
            decoder = PENetDecoder(skip_channels, in_channels, mid_channels, out_channels,
                                  kernel_size=(3, 4, 4) if is_last_decoder else 4,
                                  stride=(1, 2, 2) if is_last_decoder else 2)
            self.decoders.append(decoder)
            block_idx -= 2

        self.out_conv = nn.Conv3d(64, self.num_classes, kernel_size=3, padding=1)

        if init_method is not None:
            self._initialize_weights(init_method, focal_pi=0.01)
        
        if len(pretrained_ckpt_fn) > 0:
            self.load_pretrained(pretrained_ckpt_fn)

    def _initialize_weights(self, init_method, gain=0.2, focal_pi=None):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    if focal_pi is not None and hasattr(m, 'is_output_head') and m.is_output_head:
                        # Focal loss prior (~0.01 prob for positive, see RetinaNet Section 4.1)
                        nn.init.constant_(m.bias, -math.log((1 - focal_pi) / focal_pi))
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) and m.affine:
                # Gamma for last GroupNorm in each residual block gets set to 0
                init_gamma = 0 if hasattr(m, 'is_last_norm') and m.is_last_norm else 1
                nn.init.constant_(m.weight, init_gamma)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.net_input_channels:
            x = x.expand(-1, self.net_input_channels // x.size(1), -1, -1, -1)

        x = self.in_conv(x)

        # Encoders
        x_skips = []
        for i, encoder in enumerate(self.encoders):
            x_skips.append(x)
            if i == 0:
                x = self.max_pool(x)
            x = encoder(x)

        # ASPP layer
        x = self.asp_pool(x)

        # Classify
        cls = None

        if self.do_classify:
            cls = self.classifier(x)

        # Segment
        x_skip = None
        for decoder in self.decoders:
            x = decoder(x, x_skip)
            if x_skips:
                x_skip = x_skips.pop()

        x = self.out_conv(x)
        seg = x.squeeze(dim=1)

        return cls, seg
    
    def extract_feat(self, x):
        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        
        if x.size(1) < self.net_input_channels:     # always be 3
            x = x.expand(-1, self.net_input_channels // x.size(1), -1, -1, -1)

        x = self.in_conv(x)
        # Encoders
        x_skips = []
        for i, encoder in enumerate(self.encoders):
            x_skips.append(x)
            if i == 0:
                x = self.max_pool(x)
            x = encoder(x)

        # ASPP layer
        x = self.asp_pool(x)
        return x

    def extract_feat_batchly(self, x, bs=16):
        n = x.size(0)
        num_batch = math.ceil(n / bs)
        feat_list = []

        for i in range(num_batch):
            input_x = x[i*bs:(i+1)*bs]
            feat_x = self.extract_feat(input_x)
            feat_list.append(feat_x)
        feat = torch.cat(feat_list, dim=0)
        assert feat.size(0) == n, "feat.size(0): {} | n: {}".format(feat.size(0), n)
        return feat

    def load_pretrained(self, ckpt_path):
        """Load parameters from a pre-trairned PENetClassifier from checkpoint at ckpt_path.
        Args:
            ckpt_path: Path to checkpoint for PENetClassifier.
        Adapted from:
            https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        """
        device = 'cuda:{}'.format(self.device) if torch.cuda.is_available() else 'cpu'
        # print (device, ckpt_path)
        pretrained_dict = torch.load(ckpt_path, map_location=device)
        model_dict = self.state_dict()
        #import pdb;pdb.set_trace()

        # Filter out unnecessary keys
        pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print (f"=> the weight {ckpt_path} is loaded into PENet...")

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]

        # Debugging info
        # util.print_err('Number of fine-tuning layers: {}'
        #                .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=True))))
        # util.print_err('Number of regular layers: {}'
        #                .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=False))))

        return optimizer_parameters


if __name__ == "__main__":
    net = PENet(pretrained_ckpt_fn="../assets/penet_ckpt.pth")
    net.eval()
    net = net.cuda()
   
    a = torch.zeros((128, 1, 24, 192, 192)).cuda()
    # bb -> Nx256x3x12x12
    with torch.no_grad():
        bb = net.extract_feat_batchly(a)
    import pdb; pdb.set_trace()















