# Chinese-Text-Classification-Pytorch
基于Pytorch实现的中文文本分类脚手架，以及常用模型对比。

## Structure
```python
├─config              # 配置文件目录
├─data                # 数据目录
├─log                 # log 目录
├─output              
│  ├─model_data       # 模型存放目录
│  └─result           # 生成结果目录
├─pretrain            # 预训练模型存放目录
├─src                 # 主要代码
│  ├─datasets         # dataset 
│  ├─models           # model
│  └─tricks       
│     ├─adversarial   # 对抗训练
│     └─loss          # 特殊 loss
└─utils               # 工具代码
    
```

## Data
使用 [THUCTC](http://thuctc.thunlp.org/) 数据集，抽取了20万条新闻标题数据。文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

| 数据集  | 数据量 |
| ------ | --- |
| 训练集 | 18万 |
| 验证集 | 1万 |
| 测试集 | 1万 |	

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

## Pre Trained

1. Pre Trained Embedding
Embedding 词向量使用的是 [sogou](链接: https://pan.baidu.com/s/1EOcbzlD4BNGHnX2MEc0CNA 提取码: zvsg 复制这段内容后打开百度网盘手机App，操作更方便哦).
2. Pre Trained Model
  * chinese_wwm_ext_pytorch
  * chinese_roberta_wwm_ext_pytorch

## Run Code

```bash
python run.py --model TextCNN --save_by_step True --patience 1000
python run.py --model TextRNN --save_by_step True --patience 1000
python run.py --model TextRCNN --save_by_step True --patience 1000
python run.py --model DPCNN --save_by_step True --patience 1000

python run.py --model BertFC --save_by_step True --patience 1000
python run.py --model BertCNN --save_by_step True --patience 1000
python run.py --model BertRNN --save_by_step True --patience 1000
python run.py --model BertRCNN --save_by_step True --patience 1000
```
  
## Result


## Note
1. 按 step 存模型比按 epoch 存模型精度高。
2. 使用 BertModel 加载不同的预训练模型，比如 Albert 可能有问题。
   使用对应预训练模型的model 加载也可能有问题，比如 Roberta。
   后续需要做实验探索，目前先使用 BertModel 加载，通过 pre_trained_model 控制加载不同的预训练模型
   如 XLNet，Albert，Roberta 等。