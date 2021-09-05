# SoftMasked Bert文本纠错模型实现  
论文：《Spelling Error Correction with Soft-Masked BERT》  
1、基于Pytorch1.9.0，Python3.8，参考了github上的诸多实现，修复了各家的bug和结构遗漏，增添了海量注释；  
2、数据集为Narts（或者叫sighan？），直接从一位兄弟的复现里复制出来的数据，想用自己的数据的话照着格式处理就行；   
3、一些参数可以去model/config.py下修改；  
4、运行model/train.py即可训练模型（测试函数也在其中），centos7可无错运行，其他平台尚未测试；
5、根目录下的main.py目前为空，计划在其中写实际运用的方法，不过这就是另一个项目了。



