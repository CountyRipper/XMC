# XMC
XMC task based on semantic gneration
## 数据处理
数据预处理
1. all label 获取
2. 从数据集中解析得到json的train和test文件
3. 理论上应该对all label提取做词干化和去重
4. eurlex-4k数据集的text已经词干化，而label没有词干化，不同的数据集处理方式会有区别，根据实际而定

## pegasus训练/BART训练
