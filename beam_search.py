# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class for generating captions from an image-to-text model.
Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import heapq



class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self,
                 embedder,
                 rnn,
                 classifier,
                 eos_id,
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):
        """Initializes the generator.

        Args:
          embedder: 词嵌入层（nn.Embedding 实例）
          rnn: 循环神经网络（nn.LSTM 实例）
          classifier: 输出层（nn.Linear 实例，预测词表概率）
          eos_id: EOS 标记的词表索引（终止符）
          beam_size: 束搜索的 beam 宽度
          max_caption_length: 生成字幕的最大长度
          length_normalization_factor: 长度归一化因子（>0 时 longer 字幕更优）
        """
        self.embedder = embedder
        self.rnn = rnn
        self.classifier = classifier
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, rnn_input, initial_state=None):
        """Runs beam search caption generation on a single image.

        Args:
          rnn_input: 图像特征输入（形状：(1, batch_size, embedding_size)）
          initial_state: RNN 初始状态（默认 None，LSTM 为 (h0, c0) 元组）

        Returns:
          list: 生成的字幕词表索引列表（按分数降序）
          list: 对应字幕的分数列表
        """

        def get_topk_words(embeddings, state):
            """辅助函数：输入嵌入和状态，输出 top-k 词、对数概率和新状态"""
            output, new_states = self.rnn(embeddings, state)
            output = self.classifier(output.squeeze(0))  # 压缩时间维度（1→移除）
            # 修复1：显式指定 dim=1（词表维度），避免 deprecated 警告
            logprobs = log_softmax(output, dim=1)
            logprobs, words = logprobs.topk(self.beam_size, 1)  # 取 top-k 词
            return words.data, logprobs.data, new_states

        # 初始化：部分字幕（未到 EOS）和完整字幕（已到 EOS）的 TopN 容器
        partial_captions = TopN(self.beam_size)
        complete_captions = TopN(self.beam_size)

        # 首次预测：用图像特征生成初始 top-k 词
        words, logprobs, new_state = get_topk_words(rnn_input, initial_state)
        for k in range(self.beam_size):
            cap = Caption(
                sentence=[words[0, k].item()],  # 转为 Python 整数（避免张量残留）
                state=new_state,
                logprob=logprobs[0, k].item(),  # 对数概率（标量）
                score=logprobs[0, k].item()     # 初始分数=对数概率
            )
            partial_captions.push(cap)

        # 迭代扩展字幕（直到最大长度或无部分字幕）
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()  # 取出当前所有部分字幕
            partial_captions.reset()  # 重置容器，准备接收新候选

            # 1. 构造批量输入（所有部分字幕的最后一个词）
            input_feed = torch.LongTensor([c.sentence[-1] for c in partial_captions_list])
            # 若模型在 GPU，输入也移到 GPU
            if rnn_input.is_cuda:
                input_feed = input_feed.cuda()

            # 修复2：移除 volatile=True，改用 torch.no_grad() 禁用梯度（现代 PyTorch 规范）
            with torch.no_grad():
                input_feed = Variable(input_feed)  # 转为 Variable（兼容旧代码，可直接用张量）

            # 2. 构造批量状态（拼接所有部分字幕的 RNN 状态）
            state_feed = [c.state for c in partial_captions_list]
            if isinstance(state_feed[0], tuple):  # LSTM 状态（h, c）元组
                state_feed_h, state_feed_c = zip(*state_feed)
                state_feed = (
                    torch.cat(state_feed_h, 1),  # 按 batch 维度拼接（1 是 batch 维度）
                    torch.cat(state_feed_c, 1)
                )
            else:  # GRU 状态（仅 h）
                state_feed = torch.cat(state_feed, 1)

            # 3. 词嵌入 + 预测 top-k 词
            embeddings = self.embedder(input_feed).view(1, len(input_feed), -1)  # 恢复时间维度（1）
            words, logprobs, new_states = get_topk_words(embeddings, state_feed)

            # 4. 扩展每个部分字幕，更新候选池
            for i, partial_caption in enumerate(partial_captions_list):
                # 提取当前字幕对应的 RNN 新状态（从批量中切片）
                if isinstance(new_states, tuple):  # LSTM 状态
                    state = (
                        new_states[0].narrow(1, i, 1),  # 取第 i 个 batch 的状态（宽度 1）
                        new_states[1].narrow(1, i, 1)
                    )
                else:  # GRU 状态
                    state = new_states.narrow(1, i, 1)

                # 遍历 top-k 词，生成新候选字幕
                for k in range(self.beam_size):
                    w = words[i, k].item()  # 当前词的索引（转为 Python 整数）
                    new_sentence = partial_caption.sentence + [w]  # 扩展字幕
                    new_logprob = partial_caption.logprob + logprobs[i, k].item()  # 累计对数概率
                    new_score = new_logprob  # 初始分数=累计对数概率

                    # 若当前词是 EOS：加入完整字幕池（需长度归一化）
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            # 长度归一化：分数 /= 长度^归一化因子
                            new_score /= len(new_sentence) ** self.length_normalization_factor
                        complete_captions.push(Caption(new_sentence, state, new_logprob, new_score))
                    # 否则：加入部分字幕池
                    else:
                        partial_captions.push(Caption(new_sentence, state, new_logprob, new_score))

            # 若部分字幕池为空（如 beam_size=1 且已到 EOS），提前终止
            if partial_captions.size() == 0:
                break

        # 若无完整字幕：用部分字幕兜底（避免无输出）
        if not complete_captions.size():
            complete_captions = partial_captions

        # 提取所有完整字幕（按分数降序排序）
        caps = complete_captions.extract(sort=True)

        # 返回字幕索引列表和对应分数列表
        return [c.sentence for c in caps], [c.score for c in caps]
