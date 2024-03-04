# 基于语言增强的图像新类别发现 (参考代码)

## Python开发环境配置

```bash
# 创建虚拟环境
conda create -n dassl python=3.7
conda activate dassl
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge # torch (version >= 1.7.1)
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装 Dassl 工具库
cd Dassl.pytorch-master/
pip install -r requirements.txt
python setup.py develop

cd ..
# 安装其他库
pip install -r requirements.txt

# 完成
```

## 数据

### 训练数据

参考 `chatglm_gen.ipynb` 文件，使用 chatglm-6b 生成各个类别文本描述。生成样例为 `ChatGLM_w2s_coco_10s.json`。

### 测试数据

下载公开的测试集，并解压到项目目录。A榜测试阶段，将解压后的文件夹 `A榜数据集` 放置到本项目根目录。

## 运行

``` bash
cd scripts/
bash main.sh pazhou_distill_chatglm rn50 end 16 False pazhou_chatglm_valid 0
```

``` python
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_caption.py [args]
```

生成分类得分文件 `impreds.json`

## 提交格式

选手需要提交两个文件。一个是.txt格式的文件，包含提交结果对应的模型下载链接，命名为readme.txt. 一个是json格式的预测结果文件（即，上一步中生成的结果 impreds.json），命名为impreds.json。json文件的结构为list类型，list中每个元素是单独一个含有80类预测分值的list类型。json文件格式如下所示。将两个文件打包为submit.zip后提交（创建submit文件夹，将两个文件放到submit中再压缩为submit.zip）。

## 实验记录

[Google Sheet - VPT Competition](https://docs.google.com/spreadsheets/d/1SD20BEqSzzIs2o_26t5JikB0mkMv9f51fx2aW2pzu10/edit?usp=sharing)