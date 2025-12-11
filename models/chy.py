import torch
import torch.nn as nn

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
    def __init__(self, input_dim=80, encoder_dim=512):
        super(ResNet50, self).__init__()
        self.inplanes = 64
        # Assuming input is (Batch, Time, Freq) -> (Batch, 1, Freq, Time)
        # We treat Freq as Height, Time as Width.
        # Input channels is 1.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, encoder_dim)

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
