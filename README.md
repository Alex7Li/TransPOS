# SemiTPOT
Semi-Supervised Twitter PoS Tagging: 11-785: Midterm Project Report

## Repository Description
This repository contains several datasets that can be used for semi-supervised learning. Ark, TPANN and TweeBank are Twitter datasets while the others contain data from other domains. Additionally, it contains several files that can be used to train and evaluate model performances on different datasets.

1. training.py
This file can be used to run an experiment where at each iteration, a model is trained on a particular dataset then evaluated on all datasets. The experiment will run all possible combinations (i.e. iterate through all models, and for each model, iterate through the datasets and train the model on each one) and save the results to a pickle file that can be used for visualization

2. plot_confusion_matrix.py
This file loads the results saved to the pickle file by training.py and creates several nxn matrices where each matrix represents how the performance of the model varied depending on the dataset that it was trained on as well as the dataset that it was evaluated on.

3. plot_label_confusion.py
This file trains a model on one dataset and evaluates it on another. It then displays a confusion matrix indicating what predictions the model made correctly on the validation dataset and which ones it got wrong.

4. pseudolabels.py
This file trains a teacher model in a labelled dataset. It is then used to create pseudolabels for another unlabelled dataset B that can be used in conjunction with the labelled dataset to train a student model. This technique is currently a work in progress.

Other than these main files, there are other utility files. These include the dataloading_utils.py and the load_{dataset_name}.py files which help to load and process datasets.

## Setup

1) Unzip the datasets
cd into the ArkDataset, TPANNDataset, and TweeBankDataset folders and run the command

```bash
unzip [filename].zip
```

2) Install the dependencies
You can try

```bash
pip install -r requirements.txt
```

If it doesn't work just install what you need by individually installing the required packages

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install numpy scipy plotly seaborn pandas matplotlib
pip3 install conllu
pip3 install transformers
```

(Maybe you need to install torch separately)

3) Run the code

## References
To view the list of references on where we obtained the datasets, models and semi-supervised learning technique, please view the references section in the report.