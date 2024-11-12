---
puppeteer:
  timeout: 3000 # <= Special config, which means waitFor 3000 ms
---

# 模型 1：CNN + RNN 运行说明

<!-- 助教尝试跑通这个模型的 Pytorch 版本，结果被成堆的 bug 恶心到了，于是决定提供自己 fork 后尝试更改过的 Pytorch 版本供参考 -->

- [TensorFlow 版本参考代码（原版）](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)
- [Pytorch 迁移版本参考代码](https://github.com/Ladbaby/project_2024_LaTeX_OCR_Pytorch)。

## 该选择 TensorFlow 版本还是 Pytorch 版本？

TensorFlow 版本的代码非常老旧，且科研人员用 Pytorch 多很多

## 如何训练？

以 Pytorch 版本为例，确保装好了一些额外的包：

```shell
pip install -r requirements.txt
```

确保 `config.py` 中的参数设置符合实际情况。

训练，启动！

```shell
python train.py
```

## 训练完了要保存什么？

最重要的：根目录下会有模型的训练结果 `.pth` 模型权重文件

最终提交要用到的：
- 模型在测试集上输出的标签
- 模型的复杂度

    单位：Mac。**测试模型复杂度时 encoder 输入维度为 (1, 1, 64, 64); decoder 输入维度为 (1, 512, 32, 32)**，即调用方式为：

    ```python
    def test_params_flop(model,x_shape):
        """
        You need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
        """
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model.cuda(), 
                x_shape, 
                as_strings=True,
                print_per_layer_stat=False
            )
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    test_params_flop(encoder, (1, 64, 64))
    test_params_flop(decoder, (512, 32, 32))
    ```
    
    本模型的原始复杂度，encoder 参考为 9.25 GMac，decoder 参考为 1.02 GMac，所以最终计算的时候按照 1.03e10 Mac (即 10.27 GMac) 的总大小来带入公式算 Model Complexity Score

    > PS: 你可能好奇为什么实际的 shape 会在第一个维度多出一个 1，因为 `get_model_complexity_info` 以 batch size 为 1 进行的测试

请尽力在不损失太多性能的情况下，尝试优化模型的结构以减小模型复杂度，以获得较好的最终成绩。

## 如何提高模型的最终表现？

评价的指标不仅有模型的精度，还有新增了模型的复杂度。所以**照抄原本的模型会导致分数很低**，各位同学还需要想办法优化模型的复杂度。

- 模型精度：

    多摸索，尝试更改 `config.py` 中的一些参数，例如：

    - `epochs`: 训练的轮数
    - `encoder_lr` / `decoder_lr`: 学习率

    实现动态学习率；以及进一步清洗数据集，把过小的图片筛选掉，都可能提高最终表现
- 模型复杂度：

    需要尝试优化模型当中的部分结构，观察模型复杂度变化和结果精度的变化，取得一个平衡
