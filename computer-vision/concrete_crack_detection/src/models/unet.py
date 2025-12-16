import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Блок Двойной Свертки (DoubleConv) ---
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Первая свертка: in_channels -> out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Вторая свертка: out_channels -> out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# --- 2. Блок Сжатия (Down) ---
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# --- 3. Блок Расширения (Up) ---
class Up(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels, out_channels):
        """
        in_channels: Общее число каналов после конкатенации (C_up + C_skip)
        out_channels: Число каналов после финальной DoubleConv (C_new_layer)
        """
        super().__init__()

        # C_in_up = Каналы, поступающие от предыдущего слоя декодера (x1)
        C_in_up = in_channels - out_channels
        # C_skip = Каналы skip-connection (x2)
        C_skip = out_channels

        # ConvTranspose2d: Вход C_in_up, Выход C_skip
        self.up = nn.ConvTranspose2d(C_in_up, C_skip, kernel_size=2, stride=2, bias=False)

        # DoubleConv принимает конкатенацию (C_out_up + C_skip) = C_skip + C_skip
        self.conv = DoubleConv(C_skip + C_skip, out_channels)


    def forward(self, x1, x2):
        """
        x1: Тензор из декодера (после предыдущего Up-блока)
        x2: Тензор из skip-connection (из Encoder)
        """

        # 1. Апскейлинг x1
        x1 = self.up(x1)

        # 2. Обработка несовпадения размеров (padding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 3. Конкатенация (x2 из Encoder, x1 из Decoder)
        x = torch.cat([x2, x1], dim=1)

        # 4. Финальная двойная свертка
        return self.conv(x)


# --- 4. Основная Архитектура U-Net (для ~7.7M) ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ИЗМЕНЕНИЕ: Каналы установлены для достижения ~7.7М параметров
        # Кодировщик (Encoder) - C_out: 32, 64, 128, 256, 512
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)  # Боттлнек

        # Декодировщик (Decoder) - C_concat=C_out_prev+C_skip, C_out_conv=C_skip
        # Skip connections: x4=256, x3=128, x2=64, x1=32

        # Up(C_in_up + C_skip, C_skip)
        self.up1 = Up(512 + 256, 256)  # C_in_up=512, C_skip=256. DoubleConv(512, 256)
        self.up2 = Up(256 + 128, 128)  # C_in_up=256, C_skip=128. DoubleConv(256, 128)
        self.up3 = Up(128 + 64, 64)  # C_in_up=128, C_skip=64. DoubleConv(128, 64)
        self.up4 = Up(64 + 32, 32)  # C_in_up=64, C_skip=32. DoubleConv(64, 32)

        # Финальный слой (1x1 свертка для получения n_classes)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Кодировщик (сохраняем x1-x4 для skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Боттлнек

        # Декодировщик
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Финальная свертка
        logits = self.outc(x)

        return logits