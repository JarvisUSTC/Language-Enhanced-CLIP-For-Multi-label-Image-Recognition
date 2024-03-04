# Language-Enhanced-CLIP-For-Multi-label-Image-Recognition

This is the **third-place solution** for the Language-Enhanced Image New Category Discovery at the 2nd [The Guangdong–Hong Kong–Macao Greater Bay Area](https://www.dsec.gov.mo/BayArea/) International Algorithm and Programming Contest (**Visual Prompt Tuning Challenge @ CVPR 2023 HIT Workshop (2023)**). 

The team members include 
[Jiawei Wang](http://home.ustc.edu.cn/~wangjiawei/about.html), Zhihang Liu, Zuan Gao, Boqiang Zhang, and Yichun Feng. 

The solution slides can be found in [`Final-defense-slides.pptx`](./Final-defense-slides.pptx).

Please fined the Chinese version of README in [README-Zh.md](./README-Zh.md).

## Error Margin
Local environment: A40 GPU server. Considering differences in local multi-card training using DDP, server single-card training, data sampling, and environmental variations, a final accuracy error margin of ±0.2 may exist.

## Prediction Performance
Estimated training time: 40~50h

Estimated inference time: 10~12h

## Environment Setup
Base environment pytorch:1.11.0-cuda11.3-cudnn8, see DockerFile for details.

## Algorithm Details

### Data Generation
#### Main Data Generation (220k High-Quality Data)

Due to the limited understanding capability of chatglm-6b-v1, to generate high-quality data, the data generation by chatglm is divided into three phases.

The first phase is generating object category combinations (see project/gen_cap/gen_compositions.py for details):

The main goal is for chatglm to select categories that may appear together in real-life scenes, e.g., humans and cars on the road, whereas elephants and airplanes do not. The prompt is as follows: 
```python
f"There are several categories of objects here [{names}]. Imagine some categories of objects may appear in a natural and real-life scene at the same time, such as human and car appearing in the road, while elephant and airplane do not appear in the same scene. Please output no more than 5 category combinations that You are very Certain that they must appear in a natural and real-life scene at the same time."
```
The second phase is generating rough Captions (see project/gen_cap/gen_caption.py for details):

Building on reasonable category combinations generated in the first phase, the caption generation in this phase has better prior knowledge, with prompts as follows:
```python
[f"Determine if you can provide some sentences describing a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply.", 
f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 5 very simple sentences that are distinguishable, concise and realistic. These sentences describe 5 different images where {names} MUST appear together. For example, A kitchen with a microwave, refrigerator, and sink.",
f"Based on the previous considerations, determine if this sentence '{caption}' is a good description of a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply."]
```
The captions' rationality is repeatedly checked during generation.

The third phase is filtering generated captions (see project/gen_cap/filtered_caption.py for details):

Even though the rationality of captions has been confirmed in the second phase, there are still captions that are "with Chinese characters," "irrelevant," or with unreasonable categories. Therefore, we first filter captions with rules, then use chatglm to filter out unreasonable category combinations. The prompt is as follows:
```python
system_prompt = "Suppose you are an image describer and I want you to help us to determine if the provided caption is a good description (need to be distinguishable, concise and realistic) of a real life scene. Try to keep sentences with verbs that indicate interaction between objects, such as hit, wear, play. If so, answer 'True', otherwise answer 'False'. For example, 'A kitchen with a microwave, refrigerator, and sink.' is 'True'. 'Gorilla waving on the moon' is 'False'."
user_prompt = """
Provided caption: {caption}.
"""
```
The final captions are available in the ChatGLM_multi_labels_filtered.json file. (Additionally, small batches of data for preliminary experiments were built during the early development phase of the main data construction code, namely ChatGLM_multi_labels_2k_v2.json, ChatGLM_multi_labels_2k_v3.json)
#### Single Label Text Generation
Considering the data distribution includes many samples composed of a single category, we iterate through 80 category combinations to construct single-label data. The following prompt template guides chatglm in generating:
```python
single_prompt = f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 10 very simple sentences that are distinguishable, concise and

 realistic. These sentences describe 10 different images where {names} MUST appear."
```
See project/gen_cap/gen_caption_single.py for details. This resulted in five single-label data files: ChatGLM_single_label_1.json, ChatGLM_single_label_2.json, ChatGLM_single_label_3.json, ChatGLM_single_label_4.json, ChatGLM_single_label_5.json.
#### General Difficult Category Text Generation
Based on early model performance analysis, by integrating small, less frequent, and hard-to-recognize categories, we compiled a common difficult category template set as follows:
```python
challenge_cls = ['parking meter', 'backpack', 'handbag', 'bench','bottle','knife', 'spoon', 'chair', 'potted plant', 'mouse', 'remote', 'microwave', 'scissors', 'hair drier', 'toothbrush','truck']
```
Based on 1. Main data generation category combinations, for each category combination, we extract elements belonging to the difficult category template set challenge_cls to form new subsets and use these subsets to generate difficult category descriptions four times, thus constructing about 80k general difficult category data. See project/gen_cap/gen_caption_challenge.py for details, resulting in captions_score_challenge.jsonl.
#### Data Generation Notes
1. Data generation commands are placed in train_1.sh, with generated data saved under ./project/output/text_result/repro_data/.

2. Data for reproducing training scripts in train_2.sh is stored under ./project/output/text_result/generated_captions/.
3. During training, some expansion data will be generated based on the generated category combinations and COCO templates.

### Prompt Fine-tuning

Main ideas:
1. Introduction of evidence prompt and winner take all regularization from DualCoOp++[1] to enhance the baseline local prompt, with negligible prompt parameter volume introduced.
2. To allow the model to explore object correlations, we compiled a conditional probability matrix M from the co-occurrence frequency in the training data, where M[i,j] = P(j|i=1). During testing, we modulated the predicted scores based on this conditional probability matrix.
3. For testing, we designed a sliding window-based multi-scale and multi-block strategy. Given CLIP model's inherent focus on single objects, multi-scale testing is an effective method without fine-tuning the CLIP model. We integrated the test results of all blocks across all scales through a simple ensemble method.
4. EMA strategy: By analyzing the performance of reference paper[2] on MS COCO using the Exponential Moving Average (EMA) method, we found that applying EMA to the model during training could improve the model's noise resistance and robustness in recognizing co-occurring and conceptually similar object categories, especially the model's ability to recognize various object categories under specific scenarios (kitchen, dining room, home) or specific super-categories (vehicles). Specifically, the following advantageous super-category clusters: 1) **Vehicles/Scenarios:** 'car', 'aeroplane', 'train', 'truck', 'boat'; 2) **Animals:** 'bird', 'dog', 'horse', 'giraffe', 'bear'; 3) **Sports Equipment:** 'snowboard', 'surfboard', 'tennis racket', 'baseball bat', 'baseball glove', 'sports ball', 'kite'; 4) **Dining/Food:** 'bottle', 'wine glass', 'cup', 'knife', 'orange', 'pizza', 'donut', 'hot dog', 'bowl'; 5) **Home Scenario:** 'bed', 'toilet', 'sofa', 'pottedplant', 'tvmonitor', 'cell phone', 'umbrella', 'book', 'clock', 'scissors'. Thus, we applied the EMA strategy in the contest to construct an ema model, utilizing its advantages in specific categories during multi-model ensemble.
5. Difficult Category Mining: 1) **Mixed Training:** After analyzing the performance of reference paper[2] with EMA strategy on MS COCO, we chose difficult categories (e.g., 'toothbrush', 'hair drier','toaster', 'tie') + some common simple categories (e.g., ‘person’) for data combination on the basis of the contest ema model for incremental training to obtain the zema model. 2) **Pure Difficult Sample Training:** Using only difficult category generated texts, based on general difficult category texts (captions_score_challenge.jsonl), we divided the poorly performing categories of reference paper[2] on MSCOCO into three groups and constructed data with COCO templates for three groups of categories to train the contest incremental training zema model separately. 3) **Difficult Category Ensemble:** During multi-model ensemble, we combined models trained on specific categories used in training.
6. Single Model Inference Strategy: 1) Global Inference: Whole image inference; 

2) Local Inference: On the input image, sliding window partitioning is performed with windows of different sizes and proportions to achieve local object inference testing.
7. Multi-Model Ensemble: 1) For the output of local inference testing, inference enhancement is performed through a combination of cross-modal similarity feature weighted and variance weighted to suppress the predictive output of poorly discriminating windows and enhance the discrimination of effective local areas; 2) Multiple models with advantages in specific categories obtained through EMA strategy and difficult category mining are combined with the average merged model for ensemble combination;

[1] Hu, Ping, et al. DualCoOp++: Fast and Effective Adaptation to Multi-Label Recognition with Limited Annotations. arXiv preprint arXiv:2308.01890.
[2] Guo Z, Dong B, Ji Z, et al. Texts as images in prompt tuning for multi-label image recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2808-2817.

### Overall Approach Introduction

As described in the algorithm details, we used a three-stage method for high-quality data generation during the data generation phase; in the prompt tuning phase, we introduced DualCoOp++'s evidence prompt method for enhancement, and also designed a new loss function based on object co-occurrence relationships. Based on the idea of batch construction of training data, we conducted incremental training or prompt fine-tuning for categories with few samples or difficult samples. Moreover, we used Exponential Moving Average (EMA) to improve the model's robustness for categories under specific scenarios. During the testing phase, we designed a sliding window-based multi-scale, multi-block, and multi-shape strategy to further enhance model performance. Furthermore, we utilized the average text feature matching test images based on CLIP's image-text alignment capability to mitigate inconsistencies between training and testing phases. Finally, we used a combination of cross-modal similarity feature weighting and variance weighting for inference enhancement during model inference fusion, effectively suppressing the predictive output of poorly discriminating windows, and integrated models with advantages in specific categories through ensemble combination to achieve the final result.

### Innovations

- Designed a three-stage framework for data generation and processing
- Introduced evidence prompt to enhance the ability to connect image-text modalities through prompts
- Devised a score modulation strategy based on object correlation relationships
- Developed a sliding window-based multi-scale, multi-block, and multi-shape strategy to focus on local details of different scales and potential objects with extreme size ratios
- In the model inference fusion stage, designed inference enhancement based on a combination of cross-modal similarity feature weighting and variance weighting,

## Training Process
Training commands detailed in train_2.sh
- Step1 Train three main models under different main data strategies for main model fusion:
  - best: 220k high-quality main data + COCO template based on category combinations + COCO template sampling based on category combinations + low co-occurrence frequency rare sample data; 
- Step2 Train model based on EMA strategy to improve multi-label classification robustness in specific scenarios, resulting in the ema model;
- Step3 Mixed training, selecting difficult categories and some common simple categories for data mixing, then training on the basis of the EMA model, the combined categories used for mixing, resulting in the zema model;
- Step4 Train purely difficult sample data
  - diff: General difficult sample data + data constructed using COCO template for [''bench', 'zebra', 'fork', 'hair drier']; 
  - diffh: General difficult sample data + data constructed using COCO template for ['backpack', 'handbag', 'apple', 'chair'];
  - difft: General difficult sample data + data constructed using COCO template for ['bicycle', 'motorbike', 'parking meter', 'frisbee', 'skateboard', 'microwave', 'refrigerator', 'toothbrush'];

Note: 1) As we achieved good results in Preliminary Round A, we continued to use the category combinations from Preliminary Round A in Final Round B. 2) COCO template: Allowed template prompts, e.g., "a photo of [CLASS]"

## Commands
1. To directly reproduce leaderboard results, first run test.sh 
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

2. Based on feedback from the Preliminary Round reproduction group, data generation reproduction takes a long time,

 and there have been no changes to the data used in the finals compared to the preliminaries. Data has been uploaded to ./project/output/text_result/generated_captions (see train_1.sh).

3. In fact, we did not retrain models during the final stage but directly used models trained in the preliminaries. Therefore, we suggest using the models you reproduced during the preliminaries for inference (refer to step 5). **If you need to reproduce training**, the process is by running train_2.sh with the commands below (after completion, it will overwrite the leaderboard reproduction models in the best_model directory)
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

4. The model testing process is conducted by running test.sh with the following commands:
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
5. **(optional) Additionally, since there was no update to model training, you can actually use the models reproduced during the preliminaries for direct inference. You would need to manually replace the models in the best_model directory with those reproduced on the preliminary server, then run test.sh as per the above command**
## Other Notes
**project/best_board_model.zip: As running train_2.sh will overwrite the leaderboard reproduction model, it is backed up here.**
