### 一个简单且可以在cpu上运行的任务
主要借鉴项目链接：https://github.com/eladhoffer/captionGen
#### 代码主要完成部分：
* 修改 data.py 使之适应flickr8k数据集（如编写 flickr8k类等）
* 对数据集进行预处理，划分验证集与测试集，具体代码见 flickr8k/statistic_split.py
* 对其余代码作出部分修改，如使之适配cpu、修改老版本的python写法、使之适配该项目数据集的代码逻辑等等

#### 主要代码部分的介绍
* data.py:负责处理数据集，将数据转换为张量并格式化输入等等
* model.py:主要负责构建代码的整体逻辑框架，ccn+lstm
* utils.py:负责日志、统计等方面的代码
* main_raw.py:主程序，负责训练并且输出最终训练好的模型
* beam_search.py:负责束搜索的代码，帮助快速检索出概率最高的输出向量
* test.py:负责对单张图片进行图片描述，做测试
