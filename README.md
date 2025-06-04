<!-- GitAds-Verify: R5KMLMR1YZLRT1F36HSCND4BGVLRLWPA -->

# Florence-2-FineTuning

Welcome to the Florence-2-FineTuning repository. This repository contains tools and scripts to fine-tune the Florence-2 model for your custom dataset. It includes functionalities for data loading, model training, and evaluating datasets. The project is maintained by pecako2001.

the model can be seen visited on the huggingface website [Florence-2 Models](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de)

A Demo can be found on
[Florence-2 Demo](https://huggingface.co/spaces/gokaygokay/Florence-2)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Creation](#dataset-creation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the necessary dependencies, it is recommend to work in a Conda environment:
```sh
conda create -n florence-2 python=3.10 -y && conda activate florence-2
```
After creating the conda environment the github can be cloned and the necessary dependencies can be installed
```sh
git clone https://github.com/pecako2001/Florence-2-FineTuning.git
cd Florence-2-FineTuning
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install packaging
pip install -r requirements.txt
```

## Usage
This repository provides a command-line interface (CLI) for training and evaluating the Florence-2 model.

### Dataset creation
if you have folder with images the script `createdataset.py` can be used to create an dataset that can be used for training. This script can be run with the following commmand
```sh
python createdataset.py
```
this will create an unique id of your images and put it into the dataset folder. There is a thickbox that can be used to keep your `question`, `question_types` and `answers` the same for each image inside this folder.  The structure of the json looks as followed
```json
{
    "questionId": "337",
    "question": "what is the date mentioned in this letter?",
    "question_types": "['handwritten' 'form']",
    "docId": "279",
    "ucsf_document_id": "xnbl0037",
    "ucsf_document_page_no": "1",
    "answers": "['1/8/93']"
}
```
reference dataset: https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA
### Arguments

The training script accepts several arguments to configure the training process. Here are the available arguments:

- `--dataset_folder`: Folder containing the dataset (default: `dataset`).
- `--split_ratio`: Train/validation split ratio (default: `0.8`).
- `--batch_size`: Batch size for training (default: `2`).
- `--num_workers`: Number of workers for data loading (default: `0`).
- `--epochs`: Number of training epochs (default: `2`).

### Dataset Preparation

Ensure your dataset is in the correct format. Each image should have a corresponding JSON file with the same name (except the extension). The JSON file should contain the following fields:

- `questionId`
- `question`
- `question_types`
- `docId`
- `ucsf_document_id`
- `ucsf_document_page_no`
- `answers`

### Training

To train the model, use the following command:

```sh
python train.py --dataset_folder <path_to_dataset> --split_ratio 0.8 --batch_size 2 --num_workers 0 --epochs 2
```

Replace `<path_to_dataset>` with the path to your dataset folder. During training after each epoch an graph is generated and saved as an image inside the `Florence-2-FineTuning` folder.

<div>
<p style="text-align:center;"><img src="images/loss_graph.png" width="600" center />
</div>

### Evaluation

The model can be evaluaded using the predefined pyton script, the task_prompt is the task the model needs to perform there are multiple different tasks: `CAPTION`, `DETAILED_CAPTION`, `MORE_DETAILED_CAPTION` etc.. the rest can be found on the Huggingface space of the Florence-2 model.
```sh
python val.py --task_prompt "DETAILED_CAPTION" --text_input "What do you see in this image?" --image_path <path_to_image>
```


## Future Work

- **Evaluation Script**: Add scripts to evaluate the model on a validation or test dataset.
- **Preprocessing Tools**: Develop tools for data augmentation and preprocessing.
- **Model Improvements**: Integrate advanced training techniques and optimizations.
- **Interactive Visualization**: Implement interactive visualization tools for model predictions and dataset inspection.
- **Documentation**: Enhance documentation with more detailed usage examples and tutorials.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes or improvements.

## License

This project is licensed under the MIT License.
