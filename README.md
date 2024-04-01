# TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection (AAAI 2024 Paper)

by Hao Sun<sup>* 1</sup>, Mingyao Zhou<sup>* 1</sup>, Wenjing Chen<sup>†2</sup>, Wei Xie<sup>†1</sup>

<sup>1</sup> Central China Normal University, <sup>2</sup> Hubei University of Technology, <sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding authors.

 [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28304)]

----------

## Prerequisites

<b>0. Clone this repo</b>

<b>1. Prepare datasets</b>

<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from Moment-DETR. 

Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB), 
extract it under '../features' directory.
You can change the data directory by modifying 'feat_root' in shell scripts under 'tr_detr/scripts/' directory.

```
tar -xf path/to/moment_detr_features.tar.gz
```


<b>TVSum</b> : Download feature files for TVSum dataset from UMT.

Download [TVSum](https://connectpolyu-my.sharepoint.com/personal/21039533r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FUMT%2Ftvsum%2Dec05ad4e%2Ezip&parent=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FUMT&ga=1) (69.1MB),
and either extract it under '../features/tvsum/' directory or change 'feat_root' in TVSum shell files under 'tr_detr/scripts/tvsum/'.


<b>2. Install dependencies.</b>
Python version 3.7 is required.

```
pip install -r requirements.txt
```

Requirements.txt also include other libraries. Will be cleaned up soon.
For anaconda setup, please refer to the official [Moment-DETR github](https://github.com/jayleicn/moment_detr).

## QVHighlights

### Training

Training with (only video) and (video + audio) can be executed by running the shell below:

```
bash tr_detr/scripts/train.sh 
bash tr_detr/scripts/train_audio.sh 
```

Best validation accuracy is yielded at the last epoch. 

### Inference Evaluation and Codalab Submission for QVHighlights

Once the model is trained, `hl_val_submission.jsonl` and `hl_test_submission.jsonl` can be yielded by running inference.sh.

```
bash tr_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash tr_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

where `direc` is the path to the saved checkpoint.
For more details for submission, check [standalone_eval/README.md](standalone_eval/README.md).



## TVSum

Training with (only video) and (video + audio) can be executed by running the shell below:

```
bash tr_detr/scripts/tvsum/train_tvsum.sh 
bash tr_detr/scripts/tvsum/train_tvsum_audio.sh 
```

Best results are stored in 'results_[domain_name]/best_metric.jsonl'.

##  Cite TR-DETR (TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection)

If you find this repository useful, please use the following entry for citation.

```
@article{Sun_Zhou_Chen_Xie_2024, title={TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection}, volume={38}, url={https://ojs.aaai.org/index.php/AAAI/article/view/28304}, DOI={10.1609/aaai.v38i5.28304}, abstractNote={Video moment retrieval (MR) and highlight detection (HD) based on natural language queries are two highly related tasks, which aim to obtain relevant moments within videos and highlight scores of each video clip. Recently, several methods have been devoted to building DETR-based networks to solve both MR and HD jointly. These methods simply add two separate task heads after multi-modal feature extraction and feature interaction, achieving good performance. Nevertheless, these approaches underutilize the reciprocal relationship between two tasks. In this paper, we propose a task-reciprocal transformer based on DETR (TR-DETR) that focuses on exploring the inherent reciprocity between MR and HD. Specifically, a local-global multi-modal alignment module is first built to align features from diverse modalities into a shared latent space. Subsequently, a visual feature refinement is designed to eliminate query-irrelevant information from visual features for modal interaction. Finally, a task cooperation module is constructed to refine the retrieval pipeline and the highlight score prediction process by utilizing the reciprocity between MR and HD. Comprehensive experiments on QVHighlights, Charades-STA and TVSum datasets demonstrate that TR-DETR outperforms existing state-of-the-art methods. Codes are available at https://github.com/mingyao1120/TR-DETR.}, number={5}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Sun, Hao and Zhou, Mingyao and Chen, Wenjing and Xie, Wei}, year={2024}, month={Mar.}, pages={4998-5007} }
```


## LICENSE

The annotation files and many parts of the implementations are borrowed Moment-DETR and QD-DETR.
Following, our codes are also under [MIT](https://opensource.org/licenses/MIT) license.
