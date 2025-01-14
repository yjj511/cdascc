import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.heads.head_selector import HeadSelector
from lib.models.heads.moe import upsample_module
from lib.utils.Gaussianlayer import Gaussianlayer
from functions import ReverseLayerF
from tools.InfoNCELoss import InfoNCELoss


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim
    # 维度从 features_dim x num_classes  降到 output_dim
    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False


class Baseline_Counter(nn.Module):
    def __init__(self, config=None, weight=200, route_size=(64, 64), device=None):
        super(Baseline_Counter, self).__init__()
        self.config = config
        self.device = device
        self.resolution_num = config.resolution_num

        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)

        self.gaussian_maximum = self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()

        if self.config.counter_type == 'withMOE':
            self.multi_counters = HeadSelector(self.config.head).get_head()
            self.counter_copy = HeadSelector(self.config.head).get_head()
            freeze_model(self.counter_copy)
            self.upsample_module = upsample_module(self.config.head)

        self.weight = weight
        self.route_size = (route_size[0] // (2 ** self.resolution_num[0]),
                           route_size[1] // (2 ** self.resolution_num[0]))
        self.label_start = self.resolution_num[0]
        self.label_end = self.resolution_num[-1] + 1

        # cadn
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(48 * 8 * 8, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.loss_domain = torch.nn.NLLLoss()

        # self.map = MultiLinearMap()

        self.map = RandomizedMultiLinearMap(3072, 64, 3072)


        # 风格对比学习
        n_prototypes = 8
        d_model = 512
        self.style_encoder = nn.Linear(48 * 8 * 8 ,8)
        self.source_bank = nn.Parameter(torch.randn((n_prototypes, d_model)), requires_grad=True)
        self.target_bank = nn.Parameter(torch.randn((n_prototypes, d_model)), requires_grad=True)
        self.infoNCELoss = InfoNCELoss()

    def forward(self, inputs, labels=None, mode='train', target_inputs=None, alpha=0):
        if self.config.counter_type == 'withMOE':
            result = {'pre_den': {}, 'gt_den': {}}
            # hrnet48的输出结果
            # torch.Size([3, 48, 192, 192]) 高分辨率  小物体
            # torch.Size([3, 96, 96, 96])
            # torch.Size([3, 192, 48, 48])
            # torch.Size([3, 384, 24, 24])  低分辨率  大物体
            in_list = self.backbone(inputs)
            # 计数头复制一份参数
            self.counter_copy.load_state_dict(self.multi_counters.state_dict())
            # 复制品不需要梯度
            freeze_model(self.counter_copy)

            # torch.Size([3, 48, 192, 192])
            # torch.Size([3, 96, 96, 96])
            # torch.Size([3, 192, 48, 48])
            # torch.Size([3, 384, 24, 24])
            in_list = in_list[self.resolution_num[0]:self.resolution_num[-1] + 1]

            # SCL 模块
            # 从左到右 分辨率从高到低
            # torch.Size([3, 1, 768, 768])
            # torch.Size([3, 1, 384, 384])
            # torch.Size([3, 1, 192, 192])
            # torch.Size([3, 1, 96, 96])
            out_list = self.upsample_module(in_list, self.multi_counters, self.counter_copy)

            # 推理直接输出backbone的特征就可以了
            if labels is None:
                return out_list

            label_list = []

            labels = labels[self.label_start:self.label_end]
            # 可训练的高斯核 由cnn构成  生成密度图  并乘上缩放因子
            for i, label in enumerate(labels):
                label_list.append(self.gaussian(label.unsqueeze(1)) * self.weight)

            # 验证不需要loss
            if mode == 'val':
                result.update({'losses': self.mse_loss(out_list[0], label_list[0])})
                result['pre_den'].update({'1': out_list[0] / self.weight})
                result['pre_den'].update({'2': out_list[-3] / self.weight})
                result['pre_den'].update({'4': out_list[-2] / self.weight})
                result['pre_den'].update({'8': out_list[-1] / self.weight})

                result['gt_den'].update({'1': label_list[0] / self.weight})
                result['gt_den'].update({'2': label_list[-3] / self.weight})
                result['gt_den'].update({'4': label_list[-2] / self.weight})
                result['gt_den'].update({'8': label_list[-1] / self.weight})
                return result

            with torch.no_grad():
                in_list2 = self.backbone(target_inputs)
                in_list2 = in_list2[self.resolution_num[0]:self.resolution_num[-1] + 1]
                out_list2 = self.upsample_module(in_list2, self.multi_counters, self.counter_copy)

            ## cdan

            # source
            source_feature = F.interpolate(in_list[0], size=(in_list[0].shape[2] // 16, in_list[0].shape[3] // 16),
                                    mode='bilinear',
                                    align_corners=False)
            s_f = source_feature.view(source_feature.shape[0], -1)
            s_g = F.interpolate(out_list[0], size=(out_list[0].shape[2] // 64, out_list[0].shape[3] // 64),
                              mode='bilinear',
                              align_corners=False)
            s_g = s_g.view(s_g.shape[0], -1)
            s_g = F.softmax(s_g, dim=1).detach()

            reverse_feature = ReverseLayerF.apply(self.map(s_f, s_g), alpha)
            domain_output = self.domain_classifier(reverse_feature)
            domain_label = torch.zeros(domain_output.shape[0])
            domain_label = domain_label.long()
            err_s_domain = self.loss_domain(domain_output, domain_label.cuda())
            # target
            target_feature = F.interpolate(in_list2[0], size=(in_list2[0].shape[2] // 16, in_list2[0].shape[3] // 16),
                                    mode='bilinear',
                                    align_corners=False)
            t_f = target_feature.view(target_feature.shape[0], -1)
            t_g = F.interpolate(out_list2[0], size=(out_list2[0].shape[2] // 64, out_list2[0].shape[3] // 64),
                              mode='bilinear',
                              align_corners=False)
            t_g = t_g.view(t_g.shape[0], -1)
            t_g = F.softmax(t_g, dim=1).detach()
            reverse_feature = ReverseLayerF.apply(self.map(t_f, t_g), alpha)
            domain_output = self.domain_classifier(reverse_feature)
            domain_label = torch.ones(domain_output.shape[0])
            domain_label = domain_label.long()
            err_t_domain = self.loss_domain(domain_output, domain_label.cuda())

            # style regularization
            source_feature = F.interpolate(in_list[0], size=(in_list[0].shape[2] // 16, in_list[0].shape[3] // 16),
                                    mode='bilinear',
                                    align_corners=False)
            source_feature = source_feature.view(source_feature.shape[0], -1)
            source_weight_vectors = self.style_encoder(source_feature)
            source_weight_vectors = F.softmax(source_weight_vectors, dim=-1)
            source_style_feature = torch.einsum('bs, sc -> bc', source_weight_vectors, self.source_bank)



            target_feature = F.interpolate(in_list2[0], size=(in_list2[0].shape[2] // 16, in_list2[0].shape[3] // 16),
                                           mode='bilinear',
                                           align_corners=False)
            target_feature = target_feature.view(target_feature.shape[0], -1)
            target_weight_vectors = self.style_encoder(target_feature)
            target_weight_vectors = F.softmax(target_weight_vectors, dim=-1)
            target_style_feature = torch.einsum('bs, sc -> bc', target_weight_vectors, self.target_bank)

            contrastive_loss = self.infoNCELoss(source_style_feature, target_style_feature)



            ####################################################
            #  loss
            # [3,1,3,3]   [3,4,3,3]
            moe_label, score_gt = self.get_moe_label(out_list, label_list, self.route_size)

            # [3,4,3,3]
            mask_gt = torch.zeros_like(score_gt)

            # scatter一般用来生成one-hot向量
            # 生成mask块   4 3 3
            # [3,4,3,3]
            # [0., 0., 0.],
            # [0., 0., 0.],
            # [0., 0., 0.]],
            #
            # [[0., 1., 1.],
            # [0., 0., 0.],
            # [0., 0., 0.]],
            #
            # [[1., 0., 0.],
            # [1., 1., 1.],
            # [0., 0., 0.]],
            #
            # [[0., 0., 0.],
            # [0., 0., 0.],
            # [1., 1., 1.]]],

            # 每个样本的mask都不一样
            if mode == 'train' or mode == 'val':
                mask_gt = mask_gt.scatter_(1, moe_label, 1)

            loss_list = []
            outputs = torch.zeros_like(out_list[0])
            label_patch = torch.zeros_like(label_list[0])

            result.update({'acc1': {'gt': 0, 'error': 0}})

            # [3,1,3,3]
            mask_add = torch.ones_like(mask_gt[:, 0].unsqueeze(1))

            for i in range(mask_gt.size(1)):

                kernel = (int(self.route_size[0] / (2 ** i)), int(self.route_size[1] / (2 ** i)))
                # 分辨率从高到底  关注的区域越来越少
                loss_mask = F.upsample_nearest(mask_add, size=(out_list[i].size()[2:]))
                hard_loss = self.mse_loss(out_list[i] * loss_mask, label_list[i] * loss_mask)
                loss_list.append(hard_loss)

                if i == 0:
                    label_patch += (label_list[0] * F.upsample_nearest(mask_gt[:, i].unsqueeze(1),
                                                                       size=(out_list[i].size()[2:])))
                    label_patch = F.unfold(label_patch, kernel, stride=kernel)
                    B_, _, L_ = label_patch.size()
                    label_patch = label_patch.transpose(2, 1).view(B_, L_, kernel[0], kernel[1])
                else:
                    gt_slice = F.unfold(label_list[i], kernel, stride=kernel)
                    B, KK, L = gt_slice.size()

                    pick_gt_idx = (moe_label.flatten(start_dim=1) == i).unsqueeze(2).unsqueeze(3)
                    gt_slice = gt_slice.transpose(2, 1).view(B, L, kernel[0], kernel[1])
                    pad_w, pad_h = (self.route_size[1] - kernel[1]) // 2, (self.route_size[0] - kernel[0]) // 2
                    gt_slice = F.pad(gt_slice, [pad_w, pad_w, pad_h, pad_h], "constant", 0.2)
                    gt_slice = (gt_slice * pick_gt_idx)
                    label_patch += gt_slice

                # 用百分比来估算有多少预测正确了
                # 低分辨率只预测残差的mask
                gt_cnt = (label_list[i] * loss_mask).sum().item() / self.weight
                pre_cnt = (out_list[i] * loss_mask).sum().item() / self.weight
                result.update({f"x{2 ** (self.resolution_num[i] + 2)}": {'gt': gt_cnt,
                                                                         'error': max(0,
                                                                                      gt_cnt - abs(gt_cnt - pre_cnt))}})
                # 分辨率越低 关注区域越小
                # 最高分辨率的分支 没有做mask
                mask_add -= mask_gt[:, i].unsqueeze(1)
            ####################################################################

            B_num, C_num, H_num, W_num = out_list[0].size()
            patch_h, patch_w = H_num // self.route_size[0], W_num // self.route_size[1]
            label_patch = label_patch.view(B_num, patch_h * patch_w, -1).transpose(1, 2)
            # 复原成原来的尺寸
            label_patch = F.fold(label_patch, output_size=(H_num, W_num), kernel_size=self.route_size,
                                 stride=self.route_size)

            if mode == 'train' or mode == 'val':
                loss = 0
                if self.config.baseline_loss:
                    loss = loss_list[0]
                else:
                    ####################################################################
                    # 作者采用的是这个分支   分辨率从高到底 loss权重下降
                    for i in range(len(self.resolution_num)):
                        # if self.config.loss_weight:
                        loss += loss_list[i] * self.config.loss_weight[i]
                        # else:
                        #     loss += loss_list[i] /(2**(i))
                    loss += (err_s_domain + err_t_domain) * 0.2 + contrastive_loss * 0.2
                ####################################################################
                for i in ['x4', 'x8', 'x16', 'x32']:
                    if i not in result.keys():
                        result.update({i: {'gt': 0, 'error': 0}})
                result.update({'moe_label': moe_label})
                result.update({'losses': torch.unsqueeze(loss, 0)})
                result.update({'err_s_domain': torch.unsqueeze(err_s_domain, 0)})
                result.update({'err_t_domain': torch.unsqueeze(err_t_domain, 0)})
                result.update({'contrastive_loss': torch.unsqueeze(contrastive_loss, 0)})
                result['pre_den'].update({'1': out_list[0] / self.weight})
                result['pre_den'].update({'8': out_list[-1] / self.weight})
                result['gt_den'].update({'1': label_patch / self.weight})
                result['gt_den'].update({'8': label_list[-1] / self.weight})

                return result

            elif mode == 'test':

                return outputs / self.weight

    def get_moe_label(self, out_list, label_list, route_size):
        """
        :param out_list: (N,resolution_num,H, W) tensor
        :param in_list:  (N,resolution_num,H, W) tensor
        :param route_size: 256
        :return:
        """
        B_num, C_num, H_num, W_num = out_list[0].size()
        patch_h, patch_w = H_num // route_size[0], W_num // route_size[1]
        errorInslice_list = []

        for i, (pre, gt) in enumerate(zip(out_list, label_list)):
            pre, gt = pre.detach(), gt.detach()
            kernel = (int(route_size[0] / (2 ** i)), int(route_size[1] / (2 ** i)))

            weight = torch.full(kernel, 1 / (kernel[0] * kernel[1])).expand(1, pre.size(1), -1, -1)
            weight = nn.Parameter(data=weight, requires_grad=False).to(self.device)

            error = (pre - gt) ** 2
            patch_mse = F.conv2d(error, weight, stride=kernel)

            weight = torch.full(kernel, 1.).expand(1, pre.size(1), -1, -1)
            weight = nn.Parameter(data=weight, requires_grad=False).to(self.device)

            # mask = (gt>0).float()
            # mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            patch_error = F.conv2d(error, weight, stride=kernel)  # (pre-gt)*(gt>0)
            fractions = F.conv2d(gt, weight, stride=kernel)

            instance_mse = patch_error / (fractions + 1e-10)

            errorInslice_list.append(patch_mse + instance_mse)

        score = torch.cat(errorInslice_list, dim=1)
        moe_label = score.argmin(dim=1, keepdim=True)

        return moe_label, score


if __name__ == "__main__":
    from mmcv import Config

    cfg_data = Config.fromfile(
        '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/configs/NWPU/hrformer_b.py')

    print(cfg_data)
    # import pdb
    # pdb.set_trace()
    model = Baseline_Counter(cfg_data.network)
    print(model)

##
