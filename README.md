# TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection (AAAI 2024 Paper)

by Hao Sun<sup>* 1</sup>, Mingyao Zhou<sup>* 1</sup>, Wenjing Chen<sup>†2</sup>, Wei Xie<sup>†1</sup>

<sup>1</sup> Central China Normal University, <sup>2</sup> Hubei University of Technology, <sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding authors.

[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28304)]

----------

## Prerequisites

### 0. Clone this repository

```
git clone https://github.com/mingyao1120/tr_detr.git
cd tr_detr
```

### 1. Prepare datasets
If any dataset link becomes invalid, you can refer to [Hugging Face](https://huggingface.co/Lonicerin) for alternative resources.
#### QVHighlights
Download the official feature files for the QVHighlights dataset from Moment-DETR. 

- Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view) (8GB) and extract it under the `../features` directory. 
- You can modify the data directory by changing the `feat_root` parameter in the shell scripts located in the `tr_detr/scripts/` directory.

```
tar -xf path/to/moment_detr_features.tar.gz
```

#### TVSum
Download the feature files for the TVSum dataset from UMT.

- Download [TVSum](https://connectpolyu-my.sharepoint.com/personal/21039533r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FUMT%2Ftvsum%2Dec05ad4e%2Ezip&parent=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FUMT&ga=1) (69.1MB) and either extract it under the `../features/tvsum/` directory or modify the `feat_root` parameter in the TVSum shell scripts located in the `tr_detr/scripts/tvsum/` directory.

### 2. Install dependencies

Python version 3.7 is required. Install dependencies using:

```
pip install -r requirements.txt
```

> Note: The `requirements.txt` includes additional libraries that may not be required. These will be cleaned up in future updates. For Anaconda setup, refer to the official [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr).

----------

## QVHighlights

### Training

You can train the model using only video features or both video and audio features:

```
bash tr_detr/scripts/train.sh   # Only video
bash tr_detr/scripts/train_audio.sh   # Video + audio
```

The best validation accuracy is achieved at the last epoch.

### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
bash tr_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash tr_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

Replace `{direc}` with the path to your saved checkpoint. For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## TVSum

### Training

Similar to QVHighlights, you can train the model on the TVSum dataset:

```
bash tr_detr/scripts/tvsum/train_tvsum.sh   # Only video
bash tr_detr/scripts/tvsum/train_tvsum_audio.sh   # Video + audio
```

The best results are saved in `results_[domain_name]/best_metric.jsonl`.

----------

## Citation

If you find this repository useful, please cite our work:

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

----------

## License

The annotation files and parts of the implementation are borrowed from Moment-DETR and QD-DETR. Consequently, our code is also released under the [MIT License](https://opensource.org/licenses/MIT).


