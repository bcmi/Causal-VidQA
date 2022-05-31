# Causal-VidQA

The Causal-VidQA dataset contains 107,600 QA pairs from the [Causal-VidQA dataset](https://arxiv.org/pdf/2205.14895.pdf). The dataset aims to facilitate deeper video understanding towards video reasoning. In detail, we present the task of Causal-VidQA, which includes four types of questions ranging from scene description (description) to evidence reasoning (explanation) and commonsense reasoning (prediction and counterfactual). For commonsense reasoning, we set up a two-step solution by answering the question and providing a proper reason.

Here is an example from our dataset and the comparison between our dataset and other VisualQA datasets.

<div align=center ><img src="./fig/example.png"/></div>
<div align=center ><strong>Example from our Causal-VidQA Dataset</strong></div>

|     Dataset    | Visual Type | Visual Source | Annotation | Description | Explanation | Prediction | Counterfactual | \#Video/Image |   \#QA  | Video Length (s) |
|:--------------:|:-----------:|:-------------:|:----------:|:-----------:|:-----------:|:----------:|:--------------:|:-------------:|:-------:|:----------------:|
|   Motivation   |    Image    |    MS COCO    |     Man    |  &#10004; |  &#10004; | &#10004; |    $\times$    |     10,191    |    -    |         -        |
|       VCR      |    Image    |   Movie Clip  |     Man    |  &#10004; |  &#10004; | &#10004; |    $\times$    |    110,000    | 290,000 |         -        |
|     MovieQA    |    Video    | Movie Stories |    Auto    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |      548      |  21,406 |        200       |
|      TVQA      |    Video    |    TV Show    |     Man    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |     21,793    | 152,545 |        76        |
|     TGIF-QA    |    Video    |      TGIF     |    Auto    |  &#10004; |   $\times$  |  $\times$  |    $\times$    |     71,741    | 165,165 |         3        |
| ActivityNet-QA |    Video    |  ActivityNet  |     Man    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |     5,800     |  58,000 |        180       |
|    Social-IQ   |    Video    |    YouTube    |     Man    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |     1,250     |  7,500  |        60        |
|     CLEVRER    |    Video    |  Game Engine  |     Man    |  &#10004; |  &#10004; | &#10004; |   &#10004;   |     20,000    | 305,280 |         5        |
|       V2C      |    Video    |    MSR-VTT    |     Man    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |     10,000    | 115,312 |        30        |
|     NExT-QA    |    Video    |   YFCC-100M   |     Man    |  &#10004; |  &#10004; |  $\times$  |    $\times$    |     5,440     |  52,044 |        44        |
|  Causal-VidQA  |    Video    |  Kinetics-700 |     Man    |  &#10004; |  &#10004; | &#10004; |   &#10004;   |     26,900    | 107,600 |         9        |

<div align=center ><strong>Comparison between our dataset and other VisualQA datasets</strong></div>

In this page, you can find the code of some SOTA VideoQA methods and the dataset for our **CVPR** conference paper.

* Jiangtong Li, Li Niu and Liqing Zhang. *From Representation to Reasoning: Towards both Evidence and Commonsense Reasoning for Video Question-Answering*. *CVPR*, 2022. [[paper link]](https://arxiv.org/pdf/2205.14895.pdf)

## Download
1. [Visual Feature](https://cloud.bcmi.sjtu.edu.cn/sharing/ZI1F0Hfd0)
2. [Text Feature](https://cloud.bcmi.sjtu.edu.cn/sharing/NeiJfafJq)
3. [Dataset Split](https://cloud.bcmi.sjtu.edu.cn/sharing/6kEtHMarE)
4. [Text annotation](https://cloud.bcmi.sjtu.edu.cn/sharing/aszEJs8VX)
5. [Original Data](https://cloud.bcmi.sjtu.edu.cn/sharing/FYDmyDwff)

## Install
Please create an env for this project using miniconda (should install [miniconda](https://docs.conda.io/en/latest/miniconda.html) first)
```
>conda create -n causal-vidqa python==3.6.12
>conda activate causal-vidqa
>git clone https://github.com/bcmi/Causal-VidQA
>pip install -r requirements.txt 
```

## Data Preparation
Please download the pre-computed features and QA annotations from [Download 1-4](##Download).
And place them in ```['data/visual_feature']```, ```['data/text_feature']```, ```['data/split']``` and ```['data/QA']```. Note that the ```Text annotation``` is package as QA.tar, you need to unpack it first before place it to ```['data/QA']```.

If you want to extract different video features and text features from our Causal-VidQA dataset, you can download the original data from [Download 5](##Download) and do whatever your want to extract features.

## Usage
Once the data is ready, you can easily run the code. First, to run these models with GloVe feature, you can directly train the B2A by:
```
>sh bash/train_glove.sh
```
Note that if you want to train the model with BERT feature, we suggest your to first load the BERT feature to sharedarray by:
```
>python dataset/load.py
```
and then train the B2A with BERT feature by:
```
>sh bash/train_bert.sh.
```
After the train shell file is conducted, you can find the the prediction file under ```['results/model_name/model_prefix.json']``` and you can evaluate the prediction results by:
```
>python eval_mc.py
```
You can also obtain the prediction by running:
```
>sh bash/eval.sh
```
The command above will load the model from  ```['experiment/model_name/model_prefix/model/best.pkl']``` and generate the prediction file.

Hint: we have release a trained [model](https://cloud.bcmi.sjtu.edu.cn/sharing/c5IKQVMrM) for ```B2A``` method, please place this the trained weight in ```['experiment/B2A/B2A/model/best.pkl']``` and then make prediction by running: 
```
>sh bash/eval.sh
```

(*The results may be slightly different depending on the environments and random seeds.*)

(*For comparison, please refer to the results in our paper.*)

## Citation
```
@InProceedings{li2022from,
    author    = {Li, Jiangtong and Niu, Li and Zhang, Liqing},
    title     = {From Representation to Reasoning: Towards both Evidence and Commonsense Reasoning for Video Question-Answering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```
## Acknowledgement
Our reproduction of the methods is mainly based on the [Next-QA](https://github.com/doc-doc/NExT-QA) and other respective official repositories, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.