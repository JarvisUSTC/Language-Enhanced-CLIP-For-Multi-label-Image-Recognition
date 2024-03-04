# Language-Enhanced-CLIP-For-Multi-label-Image-Recognition

这是第二届粤港澳大湾区国际算法算例大赛基于语言增强的图像新类别发现第三名的解决方案。
队伍成员包括[Jiawei Wang](http://home.ustc.edu.cn/~wangjiawei/about), [Zhihang Liu](), [Zuan Gao](), [Boqiang Zhang](), [Yichun Feng]().


## 误差
本地环境：A40 GPU服务器。考虑到本地采用DDP多卡训练与服务器单卡训练不同，数据采样，以及环境不同等原因可能存在±0.2的最终精度误差。

## 预测性能
预计训练耗时: 40\~50h

预计推理耗时: 10\~12h

## 环境配置
基础环境pytorch:1.11.0-cuda11.3-cudnn8，详见DockerFile。
## 算法细节

###  数据生成
#### 主体数据生成(22w高质量数据)

由于chatglm-6b-v1的理解能力不是很强，因此为了尽可能生成高质量数据，我们将chatglm的数据生成分为三个阶段。

第一阶段是生成物体类别组合：(详情见project/gen_cap/gen_compositions.py)

主要目的是让chatglm先选出可能在真实场景或者日常生活中同时出现的类别组合，prompt如下: 
```python
f"There are several categories of objects here [{names}]. Imagine some categories of objects may appear in a natural and real-life scene at the same time, such as human and car appearing in the road, while elephant and airplane do not appear in the same scene. Please output no more than 5 category combinations that You are very Certain that they must appear in a natural and real-life scene at the same time."
```
第二阶段是生成粗糙的Caption：(详情见project/gen_cap/gen_caption.py)

依赖于第一阶段生成的合理的类别组合，第二阶段的caption生成就会有一个比较好的先验知识，prompt如下:
```python
[f"Determine if you can provide some sentences describing a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply.", 
f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 5 very simple sentences that are distinguishable, concise and realistic. These sentences describe 5 different images where {names} MUST appear together. For example, A kitchen with a microwave, refrigerator, and sink.",
f"Based on the previous considerations, determine if this sentence '{caption}' is a good description of a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply."]
```
在生成caption的过程中，我们也会反复check caption的合理性。

第三阶段是过滤生成的caption: (详情见project/gen_cap/filtered_caption.py)

虽然第二阶段已经确认过caption的合理性，但是仍然存在不少的“带中文的”，“内容无关的”，以及类别不合理的caption。因此我们先用规则过滤一遍caption，然后再使用chatglm过滤不合理的类别组合。prompt如下:
```python
system_prompt = "Suppose you are an image describer and I want you to help us to determine if the provided caption is a good description (need to be distinguishable, concise and realistic) of a real life scene. Try to keep sentences with verbs that indicate interaction between objects, such as hit, wear, play. If so, answer 'True', otherwise answer 'False'. For example, 'A kitchen with a microwave, refrigerator, and sink.' is 'True'. 'Gorilla waving on the moon' is 'False'."
user_prompt = """
Provided caption: {caption}.
"""
```
最终得到ChatGLM_multi_labels_filtered.json文件中的caption数据。
（此外，在主体数据构建代码的早期开发过程中构建了规模在5k以内的小批量数据用于初步实验，分别为ChatGLM_multi_labels_2k_v2.json,ChatGLM_multi_labels_2k_v3.json)
####  单标签文本生成
考虑到数据分布中存在不少只有单类别构成的样本，我们遍历80个类别组合构建单标签数据。利用如下的prompt模板来指导chatglm生成：
```python
single_prompt = f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 10 very simple sentences that are distinguishable, concise and realistic. These sentences describe 10 different images where {names} MUST appear."
```
详情见project/gen_cap/gen_caption_single.py, 最终得到ChatGLM_single_label_1.json, ChatGLM_single_label_2.json, ChatGLM_single_label_3.json, ChatGLM_single_label_4.json, ChatGLM_single_label_5.json一共5个单标签数据文件
####  通用难类别文本生成
基于早期模型的性能分析，通过对小目标类别、频率较低类别，难识别类别的整合。我们统计出一个通用的难类别模板集合如下：
```python
challenge_cls = ['parking meter', 'backpack', 'handbag', 'bench','bottle','knife', 'spoon', 'chair', 'potted plant', 'mouse', 'remote', 'microwave', 'scissors', 'hair drier', 'toothbrush','truck']
```
基于1. 主体数据生成中的类别组合，我们对每一条类别组合，抽取出属于难类别模板集合challenge_cls的元素构成新的子集合，并利用该子类别集合循环4次生成难类别描述，由此构建通用难类别数据约8w条。详情见project/gen_cap/gen_caption_challenge.py, 最终得到captions_score_challenge.jsonl。
####  数据生成说明
1. 数据生成命令放置于train_1.sh中，生成的数据将会保存在./project/output/text_result/repro_data/下

2. 用于复现训练脚本train_2.sh的数据存放于./project/output/text_result/generated_captions/下
3. 在训练过程中会根据生成的类别组合以及COCO 模板生成一些扩充数据。

### prompt微调

主要思路：
1. 引入DualCoOp++[1]提出的evidence prompt和winner take all regularization去增强baseline的local prompt，这个方法引入的prompt参数量可忽略不计。
2. 为了让模型挖掘物体之间的correlation，我们统计了训练数据中物体的共现频率，构建了一个条件概率矩阵M，M[i,j] = P(j|i=1)，在测试阶段，我们根据这个条件概率矩阵对预测的分数进行了调制。
3. 我们在测试时，设计了基于滑动窗口的multi-scale and multi-block策略，由于clip模型的固有问题(容易关注单个物体)，所以在不微调clip模型的条件下，multi-scale testing是一个有效的方式。我们将所有scale下的所有block的测试结果通过简单的ensemble方式进行整合。
4. EMA策略：通过分析官方给出的参考论文[2] 在MS COCO上的测试表现，我们发现通过Exponential Moving Average(EMA)方法来对论文[2]的模型进行训练，可以提升模型抗噪声能力和对共现类别、同概念类别物体的识别鲁棒性。尤其是模型对特定场景（厨房、餐厅、家居）或者特定超类（交通工具）下多种物体类别的整体识别能力。具体来说包含以下具有优势的超类类别簇，1）**交通工具/场景：**'car', 'aeroplane', 'train', 'truck', 'boat'；2）**动物：**'bird', 'dog', 'horse', 'giraffe', 'bear'；3）**体育用户用具：** 'snowboard', 'surfboard', 'tennis racket', 'baseball bat', 'baseball glove', 'sports ball', 'kite'；4）**餐厅/食物：** 'bottle', 'wine glass', 'cup', 'knife', 'orange', 'pizza', 'donut', 'hot dog', 'bowl'；5) **家居场景：** 'bed', 'toilet', 'sofa', 'pottedplant', 'tvmonitor', 'cell phone', 'umbrella', 'book', 'clock', 'scissors';因此, 我们基于将EMA策略应用到比赛中构建出ema模型，并在多模型集成中利用ema模型的优势类别。
5. 难类别挖掘: 1）**混合训练:** 分析参考论文[2]使用EMA策略在MS COCO上的表现后，选择难类别(eg. 'toothbrush', 'hair drier','toaster', 'tie')+部分简单常见类别(eg. ‘person’)进行数据组合在上述比赛ema模型原本基础上进行增量训练得到zema模型。 2）**完全难样本训练:** 只使用难类别生成的文本，在通用难类别文本(captions_score_challenge.jsonl)基础上，对进行第一步增量训练的参考论文[2]模型在MSCOCO上表现较差的类别随机分成三组后利用COCO模板三组类别分别进行数据构造并分别训练比赛中增量训练的zema模型。3）**难类别集成：** 在多模型集成阶段会根据相应用于训练的特定类别进行集成。
6. 单模型推理策略：1)全局推理: 整图推理; 2) 局部推理: 在输入图像上按照不同尺度的窗口大小和不同比例的窗口形状，进行分块后的滑窗，以实现局部目标的推理测试。
7. 多模型集成：1) 对于局部推理测试的输出，基于跨模态相似度特征加权和方差加权的组合模式进行推理增强，来抑制了分辨能力较差窗口的预测输出并增强有效局部区域的辨别能力; 2) 利用EMA策略和难类别挖掘获取的多个在特定类别上具有优势的模型与平均融合后的模型进行集成组合;

[1] Hu, Ping, et al. DualCoOp++: Fast and Effective Adaptation to Multi-Label Recognition with Limited Annotations. arXiv preprint arXiv:2308.01890.
[2] Guo Z, Dong B, Ji Z, et al. Texts as images in prompt tuning for multi-label image recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2808-2817.

### 整体思路介绍

如算法细节所描述，在数据生成阶段，我们使用了三阶段方法进行高质量数据生成；在prompt tuning阶段，我们引入DualCoOp++的evidence prompt方法进行增强，并且也根据物体共现关系设计了一个新的损失函数。基于分批构建训练数据的思想，我们针对少样本或者难样本类别进增量训练或对完全的难类别数据进行单独的prompt微调。此外，我们使用了Exponential Moving Average(EMA)来提升模型对特定场景下类别的鲁棒性。在测试阶段，我们设计了一种基于滑动窗口的multi-scale，multi-block以及multi-shape策略去进一步增强模型性能。进一步地，我们基于CLIP的图文对齐能力，利用和测试图像文本匹配的平均文本特征来缓解训练和测试阶段不一致。最终我们基于跨模态相似度特征加权和方差加权的组合模式进行推理增强，有效抑制了分辨能力较差窗口的预测输出，并利用优势类别模型集成组合的方式来整合所有的模型得到最终的结果。

### 方法的创新点

- 设计了基于三阶段的数据生成和处理框架
- 设计了evidece prompt来增强prompt连接图像-文本模态的能力
- 设计了根据物体correlation关系进行分数调制的策略
- 设计了基于滑动窗口的multi-scale , multi-block以及multi-shape策略，来关注不同尺度的局部细节以及尺寸比例极端的潜在物体
- 在模型推理融合阶段，设计了基于跨模态相似度特征加权和方差加权的组合模式进行推理增强，

## 训练流程
训练命令详见train_2.sh
- Step1 训练三种不同主体数据策略下的主模型用于主模型融合:
  - best:  22w高质量主体数据+基于类别组合的COCO模板+基于类别组合采样的COCO模板+共现频率低的少样本数据 ; 
- Step2 基于EMA策略训练模型以提升模型在特定场景下多标签分类鲁棒性，得到模型ema;
- Step3 混合训练，选择难类别和部分简单常见类别进行数据混合后，在上述EMA模型基础上训练，混合后用于的类别组合为，最终得到模型记为zema;
- Step4 单独训练完全难样本数据
- - diff:  通用难样本数据+利用COCO模板对[''bench', 'zebra', 'fork', 'hair drier']构建的数据 ; 
  - diffh:  通用难样本数据+利用COCO模板对['backpack', 'handbag', 'apple', 'chair']构建的数据;
  - difft:  通用难样本数据+利用COCO模板对['bicycle', 'motorbike', 'parking meter', 'frisbee', 'skateboard', 'microwave', 'refrigerator', 'toothbrush']构建的数据;

说明: 1）由于在A榜提交中取得了良好成绩，我们在B榜提交中延续了A榜中使用的类别组合。 2）COCO模板：允许使用的模板prompt，例如“a photo of [CLASS]"

## 命令
1. 直接复现榜单成绩时首先运行test.sh 
```bash
team=/data/PZxlbfx-03/03_SparkSquad/visual_prompt_tuning_reproduce_code
official_root=/data
docker run -it --rm --gpus="device=0" \
-v $official_root/official_model/:/workspace/official_model/ \
-v $team/project/:/workspace/project/ \
-v $official_root/official_b/:/workspace/official_b/ \
-v $team/test.sh:/workspace/test.sh \
--shm-size=32g --init 03_sparksquad /bin/bash test.sh
```

2. 根据初赛复现组的反馈，数据生成复现时间较久，且决赛与初赛所用的数据没有任何变化。数据已经上传于 ./project/output/text_result/generated_captions（参见train_1.sh）。

3. 事实上，在决赛阶段我们并没有重新训练模型，而是直接沿用了初赛训好的模型，因此我们建议直接用初赛您复现训练好的模型来完成推理（参考第5步）。**若需复现训练**，流程通过按照下方命令运行train_2.sh（运行结束后会覆盖掉best_model下的榜单复现模型）
```bash
team=/data/PZxlbfx-03/03_SparkSquad/visual_prompt_tuning_reproduce_code
official_root=/data
docker run -it --rm --gpus="device=0" \
-v $official_root/official_model/:/workspace/official_model/ \
-v $team/project/:/workspace/project/ \
-v $official_root/official_b/:/workspace/official_b/ \
-v $team/train_2.sh:/workspace/train_2.sh \
--shm-size=32g --init 03_sparksquad /bin/bash train_2.sh
```

4. 模型测试的流程通过按照下方命令运行test.sh 
```bash
team=/data/PZxlbfx-03/03_SparkSquad/visual_prompt_tuning_reproduce_code
official_root=/data
docker run -it --rm --gpus="device=0" \
-v $official_root/official_model/:/workspace/official_model/ \
-v $team/project/:/workspace/project/ \
-v $official_root/official_b/:/workspace/official_b/ \
-v $team/test.sh:/workspace/test.sh \
--shm-size=32g --init 03_sparksquad /bin/bash test.sh
```
5. **（optional） 此外，由于没有更新模型的训练，实际上可以利用初赛复现的模型直接进行推理 。需要您手动将初赛服务器上复现的模型来替换best_model目录下的模型，再按上述命令运行test.sh**
## 其他注意事项
**project/best_board_model.zip: 由于运行train_2.sh会覆盖榜单复现模型，因此在此处进行备份。**
