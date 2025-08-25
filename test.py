import torch
import matplotlib.pyplot as plt  # 修正matplotlib导入方式
import numpy as np
from PIL import Image
from model import CaptionModel  # 导入你的自定义模型类
import torch.serialization  # 导入序列化模块

# 将自定义模型类添加到安全全局列表（解决PyTorch 2.6+的安全限制）
torch.serialization.add_safe_globals([CaptionModel])

# 加载模型（使用安全模式）
model = torch.load(
    './results/2025-08-21_17-16-55/checkpoint_epoch_1.pth.tar',
    weights_only=False  # 允许加载完整模型（因为包含自定义类实例）
)['model']

# 如果模型训练时用了GPU，推理时在CPU运行需要转换设备
model = model.cpu()
model.eval()  # 设置为评估模式


def show_and_tell(filename='./flickr8k/images/split_images/val/23445819_3a458716c1.jpg', beam_size=3):
    try:
        # 读取图片
        img = Image.open(filename, 'r')
        if img.mode != 'RGB':
            img = img.convert('RGB')  # 确保图片是RGB格式

        # 显示图片
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.show()

        # 生成描述（注意：可能需要对图片进行预处理，与训练时保持一致）
        with torch.no_grad():  # 关闭梯度计算，节省内存
            captions = model.generate(img, beam_size=beam_size)
        print("生成的描述：", captions)

    except FileNotFoundError:
        print(f"错误：找不到图片文件 {filename}")
    except Exception as e:
        print(f"生成描述时出错：{str(e)}")


# 调用函数生成描述
show_and_tell()
