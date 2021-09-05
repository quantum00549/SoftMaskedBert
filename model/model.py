"""
@File : model.py
@Description: softmasked-bert文本纠错模型
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/8/30
@IDE: Pycharm Professional
@REFERENCE: 论文：《Spelling Error Correction with Soft-Masked BERT》，
            模型构建参考自https://github.com/will-wiki/softmasked-bert，
            修改了部分代码，修改了项目架构，修复了部分bug，补充了一些缺失的模型结构，在loss和准确率计算中加入了mask，并增添了海量注释。
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class bert:
    """
    需要用到的bert相关组件
    """
    def __init__(self, config):
        """
        初始化
        Args:
            config: 实例化的参数管理器
        """
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        # 加载预训练的模型
        self.embedding = self.bert.embeddings # 实例化BertEmbeddings类
        self.bert_encoder = self.bert.encoder
        # 实例化BertEncoder类，即attention结构，默认num_hidden_layers=12，也可以去本地bert模型的config.json文件里修改
        # 论文里也是12，实际运用时有需要再改
        # 查了源码，BertModel这个类还有BertEmbeddings、BertEncoder、BertPooler属性，在此之前我想获得bert embeddings都是直接用BertModel的call方法的，学习了
        self.tokenizer = BertTokenizer.from_pretrained(self.config.vocab_file)  # 加载tokenizer
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))
        # 加载[mask]字符对应的编码，并计算其embedding
        self.vocab_size = self.tokenizer.vocab_size  # 词汇量


class biGruDetector(nn.Module):
    """
    论文中的检测器
    """
    def __init__(self, input_size, hidden_size, num_layer=1):
        """
        类初始化
        Args:
            input_size: embedding维度
            hidden_size: gru的隐层维度
            num_layer: gru层数
        """
        super(biGruDetector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        # GRU层
        self.linear = nn.Linear(hidden_size * 2, 1)
        # 线性层
        # 因为双向GRU，所以输入维度是hidden_size*2；因为只需要输出个概率，所以第二个维度是1

    def forward(self, inp):
        """
        类call方法的覆盖
        Args:
            inp: 输入数据，embedding之后的！形如[batch_size,sequence_length,embedding_size]

        Returns:
            模型输出
        """
        rnn_output, _ = self.rnn(inp)
        # rnn输出output和最后的hidden state，这里只需要output；
        # 在batch_first设为True时，shape为（batch_size,sequence_length,2*hidden_size）;
        # 因为是双向的，所以最后一个维度是2*hidden_size。
        output = nn.Sigmoid()(self.linear(rnn_output))
        # sigmoid函数，没啥好说的，论文里就是这个结构
        return output
        # output维度是[batch_size, sequence_length, 1]


class softMaskedBert(nn.Module):
    """
    softmasked bert模型
    """
    def __init__(self, config, **kwargs):
        """
        类初始化
        Args:
            config: 实例化的参数管理器
        """
        super(softMaskedBert, self).__init__()
        self.config = config  # 加载参数管理器
        self.vocab_size = kwargs['vocab_size']
        self.masked_e = kwargs['masked_e']
        self.bert_encoder = kwargs['bert_encoder']

        self.linear = nn.Linear(self.config.embedding_size, self.vocab_size)  # 线性层，没啥好说的
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # LogSoftmax就是对softmax取log

    def forward(self, bert_embedding, p, input_mask=None):
        """
        call方法
        Args:
            bert_embedding: 输入序列的bert_embedding
            p: 检测器的输出，表示输入序列对应位置的字符错误概率，维度：[batch_size, sequence_length, 1]
            input_mask: extended_attention_mask，不是单纯的输入序列的mask，具体使用方法见下面的代码注释
        Returns:
            模型输出，经过了softmax和log，维度[batch_size,sequence_length,num_vocabulary]
        """
        soft_bert_embedding = p * self.masked_e + (1 - p) * bert_embedding  # detector输出和[mask]的embedding加权求和
        bert_out = self.bert_encoder(hidden_states=soft_bert_embedding, attention_mask=input_mask)
        # 之后再经过一个encoder结构
        # 这里有个大坑，原本看transformer的手册，BertModel的attention mask是用于输入的遮挡，维度是[batch_size,sequence_length]，但是这么输入肯定报错，
        # 查源码得知，encoder使用这个参数的时候做了处理（主要是因为多头注意力机制），直接在encoder模块里传入最初的mask就会报错
        # 源代码备注：
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 不能直接传入维度是[batch_size,sequence_length]的mask！具体处理方案见train.py
        h = bert_out[0] + bert_embedding  # 残差
        out = self.log_softmax(self.linear(h))  # 线性层，再softmax输出
        # out维度：[batch_size,sequence_length,num_vocabulary]
        return out

