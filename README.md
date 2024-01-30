# README

## Setup

本次实验使用到的库包含:

```
pandas==2.0.3
Pillow==9.4.0
Pillow==10.2.0
scikit_learn==1.2.2
torch==2.1.1+cu121
torchvision==0.16.1+cu121
```

使用以下命令进行安装：

```shell
pip install -r requirements.txt
```



## 文件结构

```
MultiModal_Learning
├── 实验五数据
│  ├──test_without_label.txt
│  ├──train.txt
│  │  
│  └─data
│
├── README.md
├── preparation.ipynb
├── requirements.txt 
├── MML.py	
├── ImageModel.py
├── TextModel.py
├── test.csv				 
├── train.csv        
└── prediction.txt         
```



## 执行代码流程

```
python MML.py --model multimodal
```

