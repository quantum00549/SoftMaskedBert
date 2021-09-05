"""
@File : data.py
@Description: 加载数据，并进行部分预处理
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/8/31
@IDE: Pycharm Professional
@REFERENCE: 关于torch.utils.data的中文文档：
            https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/

            关于collect_fn参数的小技巧：
            https://zhuanlan.zhihu.com/p/361830892

            关于迭代式数据集和映射式数据集的区别：
            https://pytorch.apachecn.org/docs/1.4/96.html
"""

import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class myData(Dataset):
    """
    数据管理器，继承自torch的Dataset类，映射式数据集
    """
    def __init__(self, config):
        """
        初始化
        Args:
            config: 参数管理器
        """
        super(myData, self).__init__()
        self.data = pd.read_csv(config.datapath, encoding='gbk')  # 加载数据
        self.tokenizer = BertTokenizer.from_pretrained(config.vocab_file)  # 默认就会添加起止符、未知符等设置

    def __len__(self):
        """
        按照官网手册，这是必须实现的方法
        Returns:
            数据集长度
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        也是必须实现的方法
        Args:
            item: 手册指定参数，目测是int型数字

        Returns:
            一条数据，可以自定义，比如这里我就返回了输入文本id，输出文本id，错字概率列表(标签)，mask矩阵
        """
        data = self.data.iloc[item]  # pd读指定行的数
        inp_txt = data['random_text']  # 输入文本，包含错字
        out_txt = data['origin_text']  # 输出文本，完全正确
        label = [0]+[int(x) for x in data['label'].strip().split()]+[0]
        # 列表，标志着文本对错，如[0,0,0,1]，就表示第五个字符是错的
        inp_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+list(inp_txt)+['[SEP]'])
        out_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+list(out_txt)+['[SEP]'])
        # 文字转编码；
        # 直接用encode函数来编码的话，有些特殊词会基于词粒度，所以这里选择先一个字一个字划分开，再用convert_tokens_to_ids函数；
        # 因为要用convert_tokens_to_ids来编码，而不是encode，所以要手动添加起止符。
        mask = [1]*len(inp_ids)
        # 遮挡
        return torch.tensor(inp_ids), torch.tensor(out_ids), torch.tensor(label).float(), torch.tensor(mask).float()
        # transformers建议模型输入是torch.long，即int64，这里用torch.tensor转换类型之后，默认就是int64


def collect_fn(batch_data):
    """
    自定义的collect_fn函数，详见文件头注释中的REFERENCE部分
    Args:
        batch_data: 一个batch的数据

    Returns:
        填充并转tensor之后的数据
    """
    batch_inp_ids = [data[0] for data in batch_data]
    batch_out_ids = [data[1] for data in batch_data]
    batch_label = [data[2] for data in batch_data]
    batch_mask = [data[3] for data in batch_data]
    # 从batch中取数

    batch_inp_ids = pad_sequence(batch_inp_ids, batch_first=True)
    batch_out_ids = pad_sequence(batch_out_ids, batch_first=True)
    batch_label = pad_sequence(batch_label, batch_first=True)
    batch_mask = pad_sequence(batch_mask, batch_first=True)
    # 填充，默认以最大序列长度填充

    return batch_inp_ids, batch_out_ids, batch_label, batch_mask


if __name__ == '__main__':
    from config import Config

    config = Config()
    torch_dataset = myData(config)
    data_loader = DataLoader(dataset=torch_dataset, batch_size=2, collate_fn=collect_fn)
    for i in data_loader:
        print(i)
        break
