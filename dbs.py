import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class SpatialProcessor(nn.Module):
    """
    空间处理分支 (DBS-Spatial)
    n_feats: 模块的内部特征维度
    """

    def __init__(self, n_feats):
        super().__init__()
        self.lka_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=7, padding=3, groups=1, bias=False),   # groups=n_feats,
            nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=False)
        )
        # 像素注意力pa_conv
        self.pa_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(n_feats // 4, n_feats, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(n_feats)
        self.act = nn.SiLU()

    def forward(self, x):
        x_lka = self.lka_conv(x)
        pa_map = self.pa_conv(x)  #生成pa_map
        x_fused = x_lka * pa_map
        return self.act(self.bn(self.conv(x_fused))), pa_map


class FrequencyProcessor(nn.Module):
    """
    频率处理分支 (DBS-Freq)
    n_feats: 模块的内部特征维度
    """

    def __init__(self, n_feats):
        super().__init__()
        self.amplitude_processor = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats, bias=False),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(n_feats)
        self.act = nn.SiLU()

    def forward(self, x):
        original_dtype = x.dtype
        x_fft = torch.fft.rfft2(x.float(), norm='ortho')
        amplitude = x_fft.abs()
        phase = x_fft.angle()

        processor_dtype = next(self.amplitude_processor.parameters()).dtype
        amplitude_processed = self.amplitude_processor(amplitude.to(processor_dtype))

        x_fft_processed = torch.polar(amplitude_processed.float().abs(), phase)
        x_spatial_float32 = torch.fft.irfft2(x_fft_processed, s=x.shape[2:], norm='ortho')

        return self.act(self.bn(x_spatial_float32.to(original_dtype)))


class DBS(nn.Module):
    """
    DBS 模块, 集成到主干网络中
    只输出特征图，不输出pa_map，保持与标准层相同的接口。
    """

    def __init__(self, c_in=3, c_out=32, n_feats=24):
        """
        c_in: 3 (来自图像)
        c_out: YOLOv8 第0层期望的输出通道 (例如 32 for yolov8n)
        n_feats: DBS 内部特征维度 (例如 24)
        """
        super().__init__()

        # 存储参数供SGA使用
        self.n_feats = n_feats
        self.c_out = c_out

        # 1. 初始下采样卷积 (替换 YOLOv8 第0层)
        self.pre_conv = Conv(c_in, n_feats, 3, 2)  # (B, 3, H, W) -> (B, n_feats, H/2, W/2)

        # --- 双域并行处理 (在 H/2 尺度上运行) ---
        self.spatial_branch = SpatialProcessor(n_feats)
        self.freq_branch = FrequencyProcessor(n_feats)

        # 3. 融合模块
        self.fusion = Conv(n_feats * 2, c_out, 1)  # (B, n_feats*2, H/2, W/2) -> (B, c_out, H/2, W/2)

        # self.pa_map_out = None  # 存储 pa_map

    def forward(self, x):
        # 1. 初始下采样
        x_feats = self.pre_conv(x)  # (B, n_feats, H/2, W/2)

        # 2. 并行处理
        x_spatial, pa_map = self.spatial_branch(x_feats)
        x_freq = self.freq_branch(x_feats)

        # 3. 保存 pa_map (重要：在融合之前保存)
        # pa_map 尺寸: (B, n_feats, H/2, W/2)
        # self.pa_map_out = self.spatial_branch.pa_map

        # 4. 融合特征
        x_fused = self.fusion(torch.cat([x_spatial, x_freq], dim=1))  # (B, c_out, H/2, W/2)

        # 5. 只返回特征图，与标准层保持一致
        return x_fused