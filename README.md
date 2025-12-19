## 口音分类网络
目前使用的baseline为`conformer`

### 数据集
#### ST-CMDS(中文)
下载链接：https://aistudio.baidu.com/datasetdetail/104460

解压后和csv文件放一起
```
└── data_dir
    ├── ST-CMDS-20170001_1-OS
    ├── metadata_split.csv
```

### 使用方法
训练：先预处理提取特征`python ./dataloaders/preprocess_st_cmds.py --root_dir datadir`
之后`bash train.py`

测试：`bash test.py`

记得修改参数以适应你的环境

### TODO
- 在`./models` 下创建你的CNN模型并训练
- 实现 `models/custom_model.py` 中的 Encoder 部分：
  - 任务：接收 Mel 频谱图输入 `(Batch, Time, Dim)`以及时序特征长度 `(Batch)`，提取用于口音分类的高维特征。
  - 输入输出参考`ConformerEncoder`
