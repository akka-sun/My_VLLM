# Qwen3-VL-2B-Instruct 图文参考测量

2026-11-21，RTX 4090 单卡，Transformers SDPA，模型 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。

给模型输入 [红色方块测试图](raw/red-square.png)，询问物体的颜色和形状。展开后输入 85 token，输出上限 32 token。三轮均返回 `red square`，各生成 3 token，耗时约 60 ms，PyTorch 峰值分配约 4.28 GB。

这是一张合成图片的功能验证和参考测量。计时包含视觉编码、生成和输出解码，预处理在计时前完成。[原始结果](raw/vlm-reference.json)
