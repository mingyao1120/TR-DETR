# TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection (AAAI 2024 Paper)

by Hao Sun<sup>* 1</sup>, Mingyao Zhou<sup>* 1</sup>, Wenjing Chen<sup>†2</sup>, Wei Xie<sup>†1</sup>

<sup>1</sup> Central China Normal University, <sup>2</sup> Hubei University of Technology, <sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding authors.

 [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28304)]

----------

## Prerequisites

<b>0. Clone this repo</b>

<b>1. Prepare datasets</b>

<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from Moment-DETR. 

Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view) (8GB), 
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
@inproceedings{sun_zhou2024tr,
  title={Tr-detr: Task-reciprocal transformer for joint moment retrieval and highlight detection},
  author={Sun, Hao and Zhou, Mingyao and Chen, Wenjing and Xie, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4998--5007},
  year={2024}
}
```


## LICENSE

The annotation files and many parts of the implementations are borrowed Moment-DETR and QD-DETR.
Following, our codes are also under [MIT](https://opensource.org/licenses/MIT) license.
