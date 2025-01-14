import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, hist1, hist2):
        # hist1: 正样本对的向量表示，形状为 [batch_size, hidden_dim]
        # hist2: 整个批次所有样本的向量表示，形状为 [batch_size, hidden_dim]

        # 归一化
        hist1 = F.normalize(hist1, dim=1)
        hist2 = F.normalize(hist2, dim=1)

        # 计算正样本得分
        pos_scores = torch.mm(hist2, hist2.transpose(0, 1))  # 使用点积作为相似度函数

        # 计算负样本得分
        neg_scores = torch.mm(hist2, hist1.transpose(0, 1))  # 矩阵乘法



        # 应用温度缩放的softmax来获得概率分布
        probs = torch.exp(pos_scores / hist2.shape[0] / self.temperature).sum(dim=1) / (
                torch.exp(pos_scores / hist2.shape[0] / self.temperature).sum(dim=1)  + torch.exp(neg_scores / self.temperature).sum(dim=1))
        # 计算InfoNCE损失
        loss = -torch.log(probs + 1e-16).sum()  # 1e-16 用于数值稳定性
        return loss