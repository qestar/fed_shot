import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10
from torch.distributions import Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torchtext.vocab import vocab, Vectors, GloVe
from embedding.meta import RNN
from embedding.auxiliary.factory import get_embedding


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm + 1e-9)
    return out


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


# import pytorch_lightning as pl

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.ema = EMA(planes)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.ema(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        # self.eff = EfficientAttention(64)
        # self.ema = EMA(64)
        # self.stvit = StokenAttention(64, stoken_size=[8,8])

        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return (x + x.mean(dim=1, keepdim=True)) * 0.5


def resnet12(keep_prob=1.0, avg_pool=False, drop_rate=0.0, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, drop_rate=drop_rate, **kwargs)
    return model


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

        self.up_channel = up_channel = int(0.6 * channels // self.groups)
        self.low_channel = low_channel = channels // self.groups - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // 2, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // 2, kernel_size=1, bias=False)
        # UP
        self.GWC = nn.Conv2d(up_channel // 2, channels // self.groups, kernel_size=5, stride=1,
                             padding=5 // 2, groups=2)
        self.PWC1 = nn.Conv2d(up_channel // 2, channels // self.groups, kernel_size=3, bias=False, padding=1)
        # low
        self.PWC2 = nn.Conv2d(low_channel // 2, channels // self.groups - low_channel // 2, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())


        # x2 = self.conv3x3(group_x)
        # # x2 = self.GWC(group_x) + self.PWC1(group_x)

        # Split
        up, low = torch.split(group_x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)

        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # Fuse
        out = torch.cat([Y1, Y2], dim=1)

        out = self.softmax(self.agp(out))

        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)



        Y1 = Y1.reshape(b * self.groups, c // self.groups, -1)

        Y2 = Y2.reshape(b * self.groups, c // self.groups, -1)

        out1 = out1.reshape(b * self.groups, -1, 1).permute(0, 2, 1)

        out2 = out2.reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        Y_fin = Y1 + Y2


        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, Y1) + torch.matmul(x11, Y2) + torch.matmul(out1, x22) + torch.matmul(out2,
                                                                                                          x22)).reshape(
            b * self.groups, 1, h, w)
        return x * (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MLP_header(nn.Module):
    def __init__(self, ):
        super(MLP_header, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        # projection
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        print('阿里v')
        return x




class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.input_dim = input_dim
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # print(x.shape,'--------')
        x = x.view(-1, self.input_dim)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = self.conv2(out)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = out.view(-1, 16 * 5 * 5)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x





class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return x, 0, y


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############## LeNet for MNIST ###################









class ModelFed_Adp(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, total_classes, net_configs=None, args=None):
        super(ModelFed_Adp, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 18 * 18 if args.dataset == 'miniImageNet' else 16 * 5 * 5),
                                             hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'resnet12':

            if args.dataset == 'FC100':
                self.features = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2)
                # num_ftrs=2560
                num_ftrs = 640
            else:
                self.features = resnet12(avg_pool=True, drop_rate=0.1)
                # num_ftrs = 16000
                num_ftrs = 640


        # summary(self.features.to('cuda:0'), (3,32,32))
        # print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer for few
        self.few_classify = nn.Linear(num_ftrs, n_classes)

        self.all_classify = nn.Linear(out_dim, total_classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_ftrs, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)
        print(self.state_dict().keys())

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            # print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x_ori, all_classify=False):
        h = self.features(x_ori)

        # print("h before:", h)
        # print("h size:", h.size())
        ebd = h.squeeze()
        # print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        if not all_classify:
            x = self.transformer(ebd)
            y = self.few_classify(x)
        else:
            x = self.l1(ebd)
            x = F.relu(x)
            x = self.l2(x)
            y = self.all_classify(x)
        return ebd, x, y





class LSTMAtt(nn.Module):

    def __init__(self, ebd, out_dim, n_classes, total_classes, args=None):
        super(LSTMAtt, self).__init__()

        # ebd = WORDEBD(args.finetune_ebd)

        self.args = args
        if args.dataset == '20newsgroup':
            self.max_text_len = 500
        elif args.dataset == 'fewrel':
            self.max_text_len = 38
        elif args.dataset == 'huffpost':
            self.max_text_len = 44

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.input_dim = self.ebd.embedding_dim  # + self.aux.embedding_dim

        # Default settings in induction encoder
        u = args.induct_rnn_dim
        da = args.induct_att_dim

        self.rnn = RNN(self.input_dim, u, 1, True, 0.5)

        # Attention
        self.head = nn.Parameter(torch.Tensor(da, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(u * 2, da)

        self.ebd_dim = u * 2

        self.l1 = nn.Linear(self.ebd_dim, self.ebd_dim)
        self.l2 = nn.Linear(self.ebd_dim, out_dim)

        # last layer for few
        self.few_classify = nn.Linear(out_dim, n_classes)

        self.all_classify = nn.Linear(out_dim, total_classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.ebd_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

        print(self.state_dict().keys())

    def _attention(self, x, text_len):
        '''
            text:     batch, max_text_len, input_dim
            text_len: batch, max_text_len
        '''
        batch_size, max_text_len, _ = x.size()

        proj_x = torch.tanh(self.proj(x.view(batch_size * max_text_len, -1)))
        att = torch.mm(proj_x, self.head)
        att = att.view(batch_size, max_text_len, 1)  # unnormalized

        # create mask
        idxes = torch.arange(max_text_len, out=torch.cuda.LongTensor(max_text_len,
                                                                     device=text_len.device)).unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1)).bool()
        att[~mask] = float('-inf')

        # apply softmax
        att = F.softmax(att, dim=1).squeeze(2)  # batch, max_text_len

        return att

    def forward(self, data, all_classify=False):
        """
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml
            @return output: batch_size * embedding_dim
        """

        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data[:, :self.max_text_len])

        # add augmented embedding if applicable
        # aux = self.aux(data)

        # ebd = torch.cat([ebd, aux], dim=2)

        # result: batch_size, max_text_len, embedding_dim

        # apply rnn
        ebd = self.rnn(ebd, data[:, self.max_text_len])
        # result: batch_size, max_text_len, 2*rnn_dim

        # ebd=F.dropout(ebd,p=0.8, training=self.training)

        # apply attention
        alpha = self._attention(ebd, data[:, self.max_text_len])

        # aggregate
        ebd = torch.sum(ebd * alpha.unsqueeze(-1), dim=1)
        # ebd = ebd.mean(1)

        # x=F.dropout(ebd, p=0.5,training=self.training)

        # x = self.l1(ebd)
        # x = F.relu(x)
        # x = self.l2(x)

        if not all_classify:
            x = self.transformer(ebd)
            y = self.few_classify(x)
        else:
            x = self.l1(ebd)
            x = F.relu(x)
            x = self.l2(x)
            y = self.all_classify(x)
        return ebd, x, y


