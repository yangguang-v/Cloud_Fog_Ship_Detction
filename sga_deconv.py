import torch
import torch.nn as nn
import torch.nn.functional as F


# 我们不再需要 Conv，因为 SGA 是自包含的
# from ultralytics.nn.modules import Conv


class SGA(nn.Module):
    """
    高级版SGA模块 - V2
    (将 F.interpolate 替换为可学习的 nn.ConvTranspose2d)
    """

    def __init__(self, c_in_feat, reduction_ratio=4):
        super().__init__()
        self.c_in_feat = c_in_feat
        self.reduction_ratio = reduction_ratio

        # 像素关联预测（模拟超像素关联图）
        self.pixel_affinity = nn.Sequential(
            nn.Conv2d(c_in_feat, c_in_feat // reduction_ratio, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in_feat // reduction_ratio, 9, 1),  # 9个邻域关系
            nn.Softmax(dim=1)
        )

        # 区域特征聚合（模拟超像素中心计算）
        self.region_aggregation = nn.Sequential(
            nn.Conv2d(c_in_feat, c_in_feat // reduction_ratio, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in_feat // reduction_ratio, c_in_feat, 3, padding=1),
            nn.BatchNorm2d(c_in_feat),
            nn.ReLU(inplace=True)
        )

        # 区域注意力机制
        self.region_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in_feat, c_in_feat // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in_feat // reduction_ratio, c_in_feat, 1),
            nn.Sigmoid()
        )

        # --- 修改 1: 重新定义特征重建模块 ---
        # 原始代码:
        # self.feature_reconstruction = nn.Sequential(
        #     nn.Conv2d(c_in_feat, c_in_feat // reduction_ratio, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(c_in_feat // reduction_ratio, c_in_feat, 3, padding=1)
        # )

        # 新代码 (合并上采样与重建):
        self.feature_reconstruction = nn.Sequential(
            # 1. 使用转置卷积进行可学习的 2x 上采样
            # (B, C, H/2, W/2) -> (B, C_reduced, H, W)
            nn.ConvTranspose2d(c_in_feat, c_in_feat // reduction_ratio, kernel_size=2, stride=2, padding=0),  # <-- 修改
            nn.ReLU(inplace=True),
            # 2. 使用 3x3 卷积提炼特征并恢复原始通道数
            # (B, C_reduced, H, W) -> (B, C, H, W)
            nn.Conv2d(c_in_feat // reduction_ratio, c_in_feat, 3, padding=1)  # <-- 修改
        )
        # --- 结束修改 1 ---

        print(f"[SGA_Advanced_V2] 初始化完成 - 输入通道: {c_in_feat}. (使用转置卷积)")

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 预测像素关联（模拟超像素分割）
        pixel_affinity = self.pixel_affinity(x)  # (B, 9, H, W)

        # 2. 区域特征聚合（下采样模拟超像素中心）
        region_features = self.region_aggregation(x)  # (B, C, H/2, W/2)

        # 3. 区域注意力（模拟超像素重要性）
        region_attention = self.region_attention(region_features)  # (B, C, 1, 1)

        # 4. 应用区域注意力
        attended_regions = region_features * region_attention  # (B, C, H/2, W/2)

        # --- 修改 2: 使用新模块替换 interpolate ---
        # 原始代码:
        # reconstructed = F.interpolate(attended_regions, size=(H, W), mode='bilinear', align_corners=False)
        # reconstructed = self.feature_reconstruction(reconstructed)

        # 新代码 (一步完成上采样和重建):
        reconstructed = self.feature_reconstruction(attended_regions)  # (B, C, H, W) # <-- 修改
        # --- 结束修改 2 ---

        # 6. 使用像素关联进行特征融合
        main_affinity = pixel_affinity[:, 4:5]  # 使用中心关联权重
        fused_output = x * main_affinity + reconstructed * (1 - main_affinity)

        # 7. 残差连接
        output = x + fused_output

        return output

    def __str__(self):
        return f"SGA_Advanced_V2(c_in_feat={self.c_in_feat})"