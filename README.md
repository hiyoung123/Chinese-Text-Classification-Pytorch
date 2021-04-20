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
Embedding 词向量使用的是 [sogou](https://pan.baidu.com/s/1EOcbzlD4BNGHnX2MEc0CNA).
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

base 是基本的配置，使用按 step 计算 val_score。各个模型没有调参，使用的是 config 中的配置文件，所以不一定是最优效果。
简单对比一下各个 trick 对模型的影响，以及各个模型在 acc/f1 上的效果。
可以看出，还是使用预训练模型效果提升最明显。

| ACC/F1  | base | focal loss | label smooth | lookahead |
| ------  | ---  |     ---    |     ---      |    ---    |
| TextCNN | 0.9274/0.9273 |0.9252/0.9252 |0.9264/0.9264 |0.9301/0.9301 |
| TextRNN | 0.9238/0.9237 |0.9248/0.9250 |0.9210/0.9210 |0.9186/0.9185 |
| TextRCNN | 0.9284/0.9285 |0.9253/0.9254 |0.9284/0.9282 |0.9263/0.9261 |
| DPCNN | 0.9221/0.9219 |0.9233/0.9234 |0.9252/0.9252 |0.9220/0.9218 |
| BertFC | 0.9453/0.9454 |0.9454/0.9453 |0.9492/0.9492 |0.9493/0.9493 |
| BertCNN | 0.9479/0.9479 |0.9419/0.9418 |0.9489/0.9489 |0.9479/0.9479 |
| BertRNN | 0.9482/0.9482 |0.9462/0.9462 |0.9453/0.9454 |0.9486/0.9486 |
| BertRCNN | 0.9476/0.9475 |0.9476/0.9475 |0.9484/0.9483 |0.9476/0.9476 |


## Note
1. 按 step 存模型比按 epoch 存模型精度高。
2. 使用 BertModel 加载不同的预训练模型，比如 Albert 可能有问题。
   使用对应预训练模型的model 加载也可能有问题，比如 Roberta。
   后续需要做实验探索，目前先使用 BertModel 加载，通过 pre_trained_model 控制加载不同的预训练模型
   如 XLNet，Albert，Roberta 等。