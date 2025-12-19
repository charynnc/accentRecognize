import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=512, base_width=32):
        super().__init__()
        self.inplanes = base_width

        # Input: (B, T, F) -> (B, 1, F, T)
        self.conv1 = nn.Conv2d(1, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, base_width, 2)
        self.layer2 = self._make_layer(BasicBlock, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_width * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * BasicBlock.expansion, encoder_dim)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, **kwargs)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x, input_lengths=None):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, F, T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, input_lengths

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=512, base_width=32):
        super(ResNet50, self).__init__()
        self.inplanes = base_width
        # Assuming input is (Batch, Time, Freq) -> (Batch, 1, Freq, Time)
        # We treat Freq as Height, Time as Width.
        # Input channels is 1.
        self.conv1 = nn.Conv2d(1, base_width, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, base_width, 3)
        self.layer2 = self._make_layer(Bottleneck, base_width * 2, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, base_width * 4, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, base_width * 8, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * Bottleneck.expansion, encoder_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, input_lengths=None):
        # x: (batch, seq_length, dimension)
        # Permute to (batch, 1, dimension, seq_length)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if input_lengths is not None:
            # Global Average Pooling with Masking
            B, C, H_out, W_out = x.shape
            max_len = x.size(3) # This is W_out, but we need original max len to scale
            # Actually, we can just use the ratio of W_out / original_max_len
            # But we don't have original_max_len easily available here unless we stored it or passed it.
            # Wait, x input to forward is (batch, seq_length, dimension).
            # But we permuted it.
            # Let's assume input_lengths corresponds to the seq_length dimension.
            
            # We need the max length of the input batch to calculate the ratio.
            # input_lengths is a tensor of real lengths.
            # The input x (before permute) had shape[1] as max_len.
            # But we don't have access to original x shape here easily unless we check input_lengths.max()
            # But input_lengths might be just the valid lengths, max(input_lengths) <= seq_len.
            # Usually seq_len = max(input_lengths) in a batch (padded).
            
            # Let's assume max(input_lengths) is close enough to the padded length, 
            # or better, just use the max value in input_lengths as the reference for the current batch's max time.
            # BUT, if the batch was padded to a fixed length larger than max(input_lengths), this ratio is wrong.
            # However, usually collate_fn pads to max(lengths) in the batch.
            # So max(input_lengths) == seq_len.
            
            max_seq_len = input_lengths.max().float()
            
            mask = torch.zeros((B, 1, 1, W_out), device=x.device)
            valid_areas = torch.ones((B, 1), device=x.device)
            
            for i in range(B):
                # Calculate valid length in feature map
                # ratio = input_lengths[i] / max_seq_len
                # valid_l = ratio * W_out
                
                # Avoid division by zero
                if max_seq_len > 0:
                    valid_l = int(torch.round((input_lengths[i] / max_seq_len) * W_out).item())
                else:
                    valid_l = 0
                
                valid_l = max(1, min(valid_l, W_out))
                
                mask[i, :, :, :valid_l] = 1.0
                valid_areas[i] = valid_l * H_out
            
            x = x * mask
            x = x.sum(dim=(2, 3)) # Sum over H and W
            x = x / valid_areas # Average
            
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
        x = self.fc(x)

        return x, input_lengths



class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out) # Add SE attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNet18(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_dim=512,
        base_width=32,
    ):
        """
        base_width: 控制网络宽度，默认32（原版ResNet18为64），实现轻量化。
        """
        super(SEResNet18, self).__init__()
        self.inplanes = base_width
        self.base_width = base_width
        
        self.conv1 = nn.Conv2d(1, base_width, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(SEBasicBlock, base_width, 2)
        self.layer2 = self._make_layer(SEBasicBlock, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, base_width * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 改进2: 统计池化输出维度是通道数的2倍 (Mean + Std)
        self.fc = nn.Linear(base_width * 8 * SEBasicBlock.expansion * 2, encoder_dim)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_p=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, input_lengths=None):
        # x: (batch, seq_length, dimension) -> (batch, 1, dimension, seq_length)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if input_lengths is not None:
            B, C, H_out, W_out = x.shape
            max_seq_len = input_lengths.max().float()
            
            # 生成 Mask
            mask = torch.zeros((B, 1, 1, W_out), device=x.device)
            for i in range(B):
                if max_seq_len > 0:
                    valid_l = int(torch.round((input_lengths[i] / max_seq_len) * W_out).item())
                else:
                    valid_l = 0
                valid_l = max(1, min(valid_l, W_out))
                mask[i, :, :, :valid_l] = 1.0
            
            # 改进3: 统计池化 (Statistics Pooling)
            x_masked = x * mask
            valid_areas = mask.sum(dim=(2, 3)) # (B, 1)
            valid_areas = torch.clamp(valid_areas, min=1e-5)
            
            # 1. Mean
            sum_x = x_masked.sum(dim=(2, 3))
            mean = sum_x / valid_areas
            
            # 2. Std
            # Var = E[X^2] - (E[X])^2
            sum_x2 = (x_masked ** 2).sum(dim=(2, 3))
            mean_x2 = sum_x2 / valid_areas
            var = mean_x2 - (mean ** 2)
            var = torch.clamp(var, min=1e-5) # 防止数值误差导致负数
            std = torch.sqrt(var)
            
            # 拼接 Mean 和 Std
            x = torch.cat([mean, std], dim=1) # (B, C * 2)
            
        else:
            # Fallback
            mean = self.avgpool(x).flatten(1)
            std = torch.std(x, dim=(2, 3)).flatten(1)
            x = torch.cat([mean, std], dim=1)
            
        x = self.fc(x)

        return x, input_lengths



class TDSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(TDSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*3, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        mean = self.avg_pool(x).view(b, c)

        if w > 1:
            dx = (x[:,:, :, 1:] - x[:, :, :, :-1])
            tdiff = self.avg_pool(dx).view(b, c)

            win = max(2, int(w // 8))
            hop = max(1, win // 2)

            # Aggregate over frequency first, keep time: (B, C, W)
            xt = x.mean(dim=2)

            # Ensure at least one window
            if w < win:
                win = w
                hop = max(1, win // 2)

            # Build sliding windows on time axis: (B, C, N, win)
            xtw = xt.unfold(dimension=2, size=win, step=hop)
            wmean = xtw.mean(dim=3)  # (B, C, N)

            # Weight windows by their relative energy (softmax) then aggregate
            # This puts more emphasis on the most salient segments but stays smoother than max.
            weights = F.softmax(wmean, dim=2)  # (B, C, N)
            twin = (weights * wmean).sum(dim=2)  # (B, C)
        else:
            twin = x.new_zeros((b, c))
            tdiff = x.new_zeros((b, c))

        y = torch.cat([mean, twin, tdiff], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TDSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(TDSEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = TDSELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out) # Add SE attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TDSEResNet18(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_dim=512,
        base_width=32,
    ):
        """
        base_width: 控制网络宽度，默认32（原版ResNet18为64），实现轻量化。
        """
        super(TDSEResNet18, self).__init__()
        self.inplanes = base_width
        self.base_width = base_width
        
        self.conv1 = nn.Conv2d(1, base_width, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(TDSEBasicBlock, base_width, 2)
        self.layer2 = self._make_layer(TDSEBasicBlock, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(TDSEBasicBlock, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(TDSEBasicBlock, base_width * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 改进2: 统计池化输出维度是通道数的2倍 (Mean + Std)
        self.fc = nn.Linear(base_width * 8 * TDSEBasicBlock.expansion * 2, encoder_dim)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_p=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, input_lengths=None):
        # x: (batch, seq_length, dimension) -> (batch, 1, dimension, seq_length)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if input_lengths is not None:
            B, C, H_out, W_out = x.shape
            max_seq_len = input_lengths.max().float()
            
            # 生成 Mask
            mask = torch.zeros((B, 1, 1, W_out), device=x.device)
            for i in range(B):
                if max_seq_len > 0:
                    valid_l = int(torch.round((input_lengths[i] / max_seq_len) * W_out).item())
                else:
                    valid_l = 0
                valid_l = max(1, min(valid_l, W_out))
                mask[i, :, :, :valid_l] = 1.0
            
            # 改进3: 统计池化 (Statistics Pooling)
            x_masked = x * mask
            valid_areas = mask.sum(dim=(2, 3)) # (B, 1)
            valid_areas = torch.clamp(valid_areas, min=1e-5)
            
            # 1. Mean
            sum_x = x_masked.sum(dim=(2, 3))
            mean = sum_x / valid_areas
            
            # 2. Std
            # Var = E[X^2] - (E[X])^2
            sum_x2 = (x_masked ** 2).sum(dim=(2, 3))
            mean_x2 = sum_x2 / valid_areas
            var = mean_x2 - (mean ** 2)
            var = torch.clamp(var, min=1e-5) # 防止数值误差导致负数
            std = torch.sqrt(var)
            
            # 拼接 Mean 和 Std
            x = torch.cat([mean, std], dim=1) # (B, C * 2)
            
        else:
            # Fallback
            mean = self.avgpool(x).flatten(1)
            std = torch.std(x, dim=(2, 3)).flatten(1)
            x = torch.cat([mean, std], dim=1)
            
        x = self.fc(x)

        return x, input_lengths


if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, f = 2, 200, 80
    x = torch.randn(b, t, f)
    lengths = torch.tensor([200, 160])


    m2 = TDSEResNet18(input_dim=f, encoder_dim=128, base_width=32, reduction=8, dropout=0.2)
    y2, _ = m2(x, lengths)
    print("TCResNet18 out:", y2.shape)
