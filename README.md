### 口音分类网络
目前使用的baseline为`conformer`

数据库使用accentDB（https://accentdb.org/）

#### 使用方法
训练：`bash train.py`

测试：`bash test.py`

记得修改参数以适应你的环境

#### TODO
- 在`./models` 下创建你的CNN模型并训练
- 实现 `models/custom_model.py` 中的 Encoder 部分：
  - 任务：接收 Mel 频谱图输入 `(Batch, Time, Dim)`以及时序特征长度 `(Batch)`，提取用于口音分类的高维特征。
  - 输入输出参考`ConformerEncoder`
