# 轨迹查找与动量预测

## 文件结构

- 数据文件存放在`data`目录下。
- 函数库：`SFlib_I.py`、`SFlib_II.py`
- 代码文件：`FindTrack.py`、`Predict.py`

## 轨迹查找

通过分析粒子撞击事件数据（含噪声数据），识别并且重建粒子轨迹归类，并在图中展示。

- 可直接运行的脚本文件：`FindTrack.py`
- 用到的数据文件：`data/SiHits_3D_pvar_zvar_0.03_0.50_28_0.04_0.06_2000_v1.txt`

另外，作者制作了对应的UI界面，方便用户交互式地选择事件并绘制轨迹。直接运行`FindTrack_UI.py`即可使用。

## 动量预测

经由几何聚类与圆拟合提取出轨迹在 $x–y$ 平面上的曲率/圆心及拟合误差等特征，训练一个MLP回归器来预测粒子的横向动量分量`(p_x, p_y)`，并评估预测与真值的偏差分布。

- 可直接运行的脚本文件：`Predict.py`
- 用到的数据文件：`data/SiHits_3D_pvar_0.02_10000_v2.txt`
- 生成两个图片：`figure_predict_MSE.png`、`figure_predict_partII.png`，分别展示了模型在测试集上的预测效果和评估结果的直方图。

## 文献

本项目是以下文献的实现：`J. Zhi, S. Wu, J. Zhao and X. Cao, "Hybrid Algorithms for Enhanced Vertex and Track Reconstruction," 2024 6th International Communication Engineering and Cloud Computing Conference (CECCC), Chengdu, China, 2024, pp. 15-22, doi: 10.1109/CECCC62598.2024.11063558.`

地址：https://ieeexplore.ieee.org/abstract/document/11063558
