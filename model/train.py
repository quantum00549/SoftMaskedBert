"""
@File : train.py
@Description: 模型训练、推理等功能
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/9/1
@IDE: Pycharm Professional
@REFERENCE: NLLLoss:https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
            BCELoss:https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
"""
import os
import torch
import platform
import torch.nn as nn
from data import myData, collect_fn
from torch.utils.data import DataLoader, random_split
from model import softMaskedBert, biGruDetector, bert


class engines(bert):
    def __init__(self, config):
        super(engines, self).__init__(config)
        torch_dataset = myData(self.config)
        train_size = int(len(torch_dataset) * 0.7)
        test_size = len(torch_dataset) - train_size
        train_dataset, test_dataset = random_split(torch_dataset, [train_size, test_size])
        # 数据集划分

        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.config.batch_size,
                                            collate_fn=collect_fn, shuffle=True)
        self.test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.config.batch_size,
                                           collate_fn=collect_fn)
        # 创建训练和测试数据集

        self.detector_model = biGruDetector(self.config.embedding_size, self.config.hidden_size)  # 实例化检测器
        self.detector_optimizer = torch.optim.Adam(self.detector_model.parameters(), lr=self.config.lr)  # 检测器的优化器
        self.detector_criterion = nn.BCELoss()  # 检测器部分的损失，Binary CrossEntropy

        self.model = softMaskedBert(
            self.config,
            vocab_size=self.vocab_size,
            masked_e=self.masked_e.to(self.config.device),
            bert_encoder=self.bert_encoder)  # 实例化模型
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)  # 优化器
        self.criterion = nn.NLLLoss()  # 整个模型的损失，Negative Loglikelihood
        # 关于NLLLoss的手册原文：The input given through a forward call is expected to contain log-probabilities of each class，
        # 即log(softmax)。
        # 论文里用的是NLLLoss而非交叉熵

        self.gama = 0.7
        # 论文里两个模块loss的加权系数，论文里也就说了个大概，大于0.5，因为纠错器部分的学习更困难也更重要些

        self.checkpoint = {'detector_model': self.detector_model.state_dict(),
                           'detector_optimizer': self.detector_optimizer.state_dict(),
                           'model': self.model.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           'epoch': 0}
        # 初始化模型保存点
        self.start_epoch = 0  # 初始化开始训练的轮数，在断点续训时要用到

        self.device = self.config.device  # 计算设备

    def train_model(self, resume=False):
        """
        训练模型
        Args:
            resume: 是否继续训练
        """
        if resume:
            if newest_file := self.new_file(self.config.checkpoints):
                self.checkpoint = torch.load(self.config.checkpoints + '/' + newest_file)
                # 读取模型保存文件
                print('已加载模型文件')
            else:
                print('未找到保存的模型')
        self.load_model()  # 加载模型，不管是不是断点续训，这里都没什么影响，相当于读个初始化的checkpoint

        self.detector_model.to(self.device)
        self.model.to(self.device)  # 计算设备

        self.detector_model.train()
        self.model.train()  # train模式

        for epoch in range(self.start_epoch, self.config.epoch):
            detector_correct = 0  # 检测器准确数
            corrector_correct = 0  # 纠错器准确数
            total_loss = 0  # 总loss
            num_data = 0  # 总数据量，字符粒度
            for i, batch_data in enumerate(self.train_data_loader):
                batch_inp_ids, batch_out_ids, batch_labels, batch_mask = batch_data
                batch_out_ids = batch_out_ids.to(self.device)  # 选择计算设备，下同
                batch_labels = batch_labels.to(self.device)
                batch_mask = batch_mask.to(self.device)

                batch_inp_embedding = self.embedding(batch_inp_ids).to(self.config.device)  # 获取输入文本序列的embedding表示
                prob = self.detector_model(batch_inp_embedding)  # 检测器模块的输出

                detector_loss = self.detector_criterion(prob.squeeze()*batch_mask, batch_labels.float())
                # 按照官网说明，squeeze不指定dim参数的话，就去除所有维度大小是1的维度
                # 原文：Returns a tensor with all the dimensions of input of size 1 removed,
                # if dim is given, the input will be squeezed only in this dimension,
                # 原本prob维度是[batch_size, sequence_length, 1]，
                # batch_labels维度是[batch_size, sequence_length]，
                # 这样操作后，维度就相等了

                # 还有，计算loss时不考虑padding部分

                out = self.model(
                    batch_inp_embedding,
                    prob,
                    self.bert.get_extended_attention_mask(batch_mask, batch_out_ids.shape, batch_inp_ids.device))
                # 这个get_extended_attention_mask来自BertModel继承的类，
                # 官方手册里也没介绍，当然一般也不会介绍这些具体实现，看了源码之后，直接拿出来调用

                model_loss = self.criterion((out*batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]), batch_out_ids.reshape(-1))
                # 注意这里的reshape，因为NLLLoss不能正确处理batch维度，不知道为什么这么不智能
                # 用交叉熵就没这么多麻烦事，不过论文用的不是交叉熵
                # 计算loss考虑mask
                loss = self.gama * model_loss + (1 - self.gama) * detector_loss  # 联合loss
                self.optimizer.zero_grad()  # 每次迭代需梯度置零
                # 这里有个小技巧，梯度不置零可以近似获得增大batch_size的效果，以减少显存不足的限制，比如每两个batch，梯度归零一次
                # 因为tf默认执行这个操作，学习torch之后才发现的
                loss.backward(retain_graph=True)
                self.optimizer.step()

                prob = torch.round(prob)
                detector_correct = detector_correct + sum(
                    [(prob.squeeze() * batch_mask).reshape(-1)[i].equal((batch_labels * batch_mask).reshape(-1)[i])
                     for i in range(len(prob.reshape(-1)))])

                output = torch.argmax(out, dim=-1)
                # 我看的torch教程里这里用了detach()之后再进行计算，猜测是为了稳妥，不过感觉没什么必要，这里的output并不参与求导
                # 本着对代码优雅的追求，去掉了这部分操作
                corrector_correct = corrector_correct + sum(
                    [(output*batch_mask).reshape(-1)[j].equal((batch_out_ids*batch_mask).reshape(-1)[j])
                     for j in range(len(output.reshape(-1)))])
                # 计算准确率，padding部分均视为正确，下同

                total_loss += loss.item()
                num_data += sum([len(m) for m in batch_mask])
                print(f'epoch: {epoch+1}, '
                      + f'batch: {i + 1}, '
                      + f'train loss: {total_loss / (i + 1)}%, '
                      + f'train detector accuracy: {detector_correct / num_data}, '
                      + f'train corrector_accuracy: {corrector_correct / num_data}')
            if (epoch+1)%5 == 0:  # 每五轮保存一次模型
                self.save_model(epoch)
                print(f'模型已保存，epoch: {epoch}')

    def test_model(self):
        """
        测试模型，这段代码和上面的训练模型相似度较高，但是暂时还没想到足够优雅的方式重构
        """
        if newest_file := self.new_file(self.config.checkpoints):
            self.checkpoint = torch.load(self.config.checkpoints + '/' + newest_file)
            # 读取模型保存文件
            print('已加载模型文件')
        else:
            print('未找到保存的模型，预测准确性无法保证')

        self.load_model()  # 加载模型，没有模型文件也不会报错，相当于读个刚初始化的checkpoint

        self.detector_model.to(self.device)
        self.model.to(self.device)  # 计算设备

        self.detector_model.eval()
        self.model.eval()  # 推理模式

        detector_correct = 0  # 检测器准确数
        corrector_correct = 0  # 纠错器准确数
        num_data = 0  # 总数据量，字符粒度
        total_loss = 0  # 总loss
        for i, batch_data in enumerate(self.test_data_loader):
            batch_inp_ids, batch_out_ids, batch_labels, batch_mask = batch_data
            batch_out_ids = batch_out_ids.to(self.device)  # 选择计算设备，下同
            batch_labels = batch_labels.to(self.device)
            batch_mask = batch_mask.to(self.device)

            batch_inp_embedding = self.embedding(batch_inp_ids).to(self.config.device)  # 获取输入文本序列的embedding表示
            prob = self.detector_model(batch_inp_embedding)  # 检测器模块的输出
            detector_loss = self.detector_criterion(prob.squeeze() * batch_mask, batch_labels.float())

            out = self.model(batch_inp_embedding, prob,
                             self.bert.get_extended_attention_mask(batch_mask, batch_out_ids.shape,
                                                                   batch_inp_ids.device))
            model_loss = self.criterion((out * batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]),
                                        batch_out_ids.reshape(-1))
            loss = self.gama * model_loss + (1 - self.gama) * detector_loss  # 联合loss
            prob = torch.round(prob)
            detector_correct = detector_correct + sum(
                [(prob.squeeze() * batch_mask).reshape(-1)[i].equal((batch_labels * batch_mask).reshape(-1)[i])
                 for i in range(len(prob.reshape(-1)))])
            # 检测器准确数

            output = torch.argmax(out, dim=-1)
            corrector_correct = corrector_correct + sum(
                [(output * batch_mask).reshape(-1)[j].equal((batch_out_ids * batch_mask).reshape(-1)[j])
                 for j in range(len(output.reshape(-1)))])
            # 模型准确数

            total_loss += loss.item()
            num_data += sum([len(m) for m in batch_mask])
        print(f'test loss: {total_loss / (i + 1)}%, '
              + f'train detector accuracy: {detector_correct / num_data}, '
              + f'train corrector_accuracy: {corrector_correct / num_data}')

    def load_model(self):
        """
        加载模型
        """
        self.detector_model.load_state_dict(self.checkpoint['detector_model'])  # 加载检测器模型参数
        self.detector_optimizer.load_state_dict(self.checkpoint['detector_optimizer'])  # 加载检测器优化器参数
        self.model.load_state_dict(self.checkpoint['model'])  # 加载模型参数
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])  # 加载优化器参数
        self.start_epoch = self.checkpoint['epoch']  # 设置开始的epoch

    def save_model(self, epoch):
        """
        保存模型
        Args:
            epoch: 整型数字，表示训练到第几轮
        """
        if not os.path.isdir("./checkpoints"):
            os.mkdir("./checkpoints")
        self.checkpoint['epoch'] = epoch
        torch.save(self.checkpoint, self.config.checkpoints + f'/skpt_epoch{epoch}.pt')

    @staticmethod
    def new_file(tardir):
        """
        获取指定目录下最新创建的文件，用于加载最新的模型保存点
        Args:
            tardir: 字符串，指定目录
        Returns:
            字符串，最新创建的文件名，不包含上级目录
        """
        filelist = os.listdir(tardir)  # 列出目录下所有的文件
        if not platform.system() == 'Windows':  # 非win系统应该都是反斜杠分隔符，当然也可能不对，有错误可以改改
            filelist.sort(key=lambda fn: os.path.getmtime(tardir + '/' + fn))  # 对文件修改时间进行升序排列
        else:
            filelist.sort(key=lambda fn: os.path.getmtime(tardir + '\\\\' + fn))  # win下系统默认的地址分割符是双斜杠，注：斜杠本身是转义符
        try:
            return filelist[-1]
        except IndexError:  # 如果目录下没有文件，filelist[-1]就会报IndexError
            return None


if __name__ == '__main__':
    from config import Config

    config = Config()
    handler = engines(config)
    handler.train_model(True)
    # handler.test_model()