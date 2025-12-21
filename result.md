### ST_CMDS
#### 2000
1. conformer
```
Test Loss: 0.5463 Acc: 82.94%

Per-class Accuracy:
Class 四川: 90.00%
Class 天津: 75.50%
Class 安徽: 79.50%
Class 山东: 89.00%
Class 广西: 95.00%
Class 河南: 73.00%
Class 甘肃: 70.00%
Class 黑龙江: 91.50%
```

2. ResNet50
```
Test Loss: 0.5478 Acc: 82.88%

Per-class Accuracy:
Class 四川: 84.00%
Class 天津: 75.00%
Class 安徽: 87.00%
Class 山东: 84.00%
Class 广西: 92.50%
Class 河南: 78.00%
Class 甘肃: 76.00%
Class 黑龙江: 86.50%
```

3. SEResNet18
```
Test Loss: 0.6734 Acc: 80.00%

Per-class Accuracy:
Class 四川: 95.50%
Class 天津: 68.50%
Class 安徽: 65.00%
Class 山东: 84.50%
Class 广西: 93.00%
Class 河南: 76.50%
Class 甘肃: 67.00%
Class 黑龙江: 90.00%
```

4. SEResNet18+tdiff
```
Test Loss: 0.5382 Acc: 83.31%

Per-class Accuracy:
Class 四川: 90.00%
Class 天津: 74.50%
Class 安徽: 71.00%
Class 山东: 89.00%
Class 广西: 95.00%
Class 河南: 81.00%
Class 甘肃: 82.00%
Class 黑龙江: 84.00%
```

5. SEResNet18+tcontract
```
Test Loss: 0.5573 Acc: 83.19%

Per-class Accuracy:
Class 四川: 92.50%
Class 天津: 67.00%
Class 安徽: 85.50%
Class 山东: 87.50%
Class 广西: 88.50%
Class 河南: 81.00%
Class 甘肃: 85.50%
Class 黑龙江: 78.00%
```

6. TDSEResNet18(tdiff+tcontract)
```
Test Loss: 0.5872 Acc: 83.06%

Per-class Accuracy:
Class 四川: 96.00%
Class 天津: 81.50%
Class 安徽: 84.50%
Class 山东: 85.00%
Class 广西: 84.00%
Class 河南: 66.50%
Class 甘肃: 78.00%
Class 黑龙江: 89.00%
```


7. 加拼音
```
Test Loss: 0.4113 Acc: 87.75%

Per-class Accuracy:
Class 四川: 96.50%
Class 天津: 88.50%
Class 安徽: 90.50%
Class 山东: 87.00%
Class 广西: 95.50%
Class 河南: 72.00%
Class 甘肃: 74.00%
Class 黑龙江: 98.00%
```
韵母声调分开
```
Test Loss: 0.3587 Acc: 90.44%

Per-class Accuracy:
Class 四川: 92.50%
Class 天津: 78.00%
Class 安徽: 94.50%
Class 山东: 94.00%
Class 广西: 96.50%
Class 河南: 84.00%
Class 甘肃: 86.50%
Class 黑龙江: 97.50%
```

#### 1000
ResNet50
```
Test Loss: 1.1911 Acc: 63.50%

Per-class Accuracy:
Class 四川: 74.00%
Class 天津: 53.00%
Class 安徽: 66.00%
Class 山东: 60.00%
Class 广西: 78.00%
Class 河南: 48.00%
Class 甘肃: 55.00%
Class 黑龙江: 74.00%
```

ResNet50+loss
```
Test Loss: 0.9874 Acc: 67.25%

Per-class Accuracy:
Class 四川: 75.00%
Class 天津: 56.00%
Class 安徽: 78.00%
Class 山东: 66.00%
Class 广西: 85.00%
Class 河南: 60.00%
Class 甘肃: 59.00%
Class 黑龙江: 59.00%
```

ResNet18+loss
```
Test Loss: 1.2408 Acc: 62.12%

Per-class Accuracy:
Class 四川: 77.00%
Class 天津: 40.00%
Class 安徽: 73.00%
Class 山东: 62.00%
Class 广西: 69.00%
Class 河南: 57.00%
Class 甘肃: 44.00%
Class 黑龙江: 75.00%
```

18+weight+loss
```
Test Loss: 0.9555 Acc: 66.38%

Per-class Accuracy:
Class 四川: 66.00%
Class 天津: 49.00%
Class 安徽: 77.00%
Class 山东: 65.00%
Class 广西: 82.00%
Class 河南: 47.00%
Class 甘肃: 58.00%
Class 黑龙江: 87.00%
```


