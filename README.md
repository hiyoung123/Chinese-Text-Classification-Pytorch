# Chinese-Text-Classification-Pytorch_V2
基于Pytorch实现的中文文本分类脚手架，以及常用模型对比。

## Model

1. TextRNN
2. TextCNN
3. TextRCNN
4. FastText
5. HAN
6. Bert
7. Albert
8. RoBerta
9. NeZha
10. MacBERT

## Trick

1. Flooding
2. FGM
3. Focal Loss
4. Dice Loss
5. Label Smooth
6. scheduler
7. max_gradient_norm
8. fp16
9. lookahead
10. MSD
11. 初始化权重

## 代码结构

## Note
1. 需要在全局设置 seed。
2. 按 step 存模型比按 epoch 存模型精度高。
3. 使用 BertModel 加载不同的预训练模型，比如 Albert 可能有问题。
   使用对应预训练模型的model 加载也可能有问题，比如 Roberta。
   后续需要做实验探索，目前先使用 BertModel 加载，通过 pre_trained_model 控制加载不同的预训练模型
   如 XLNet，Albert，Roberta 等。