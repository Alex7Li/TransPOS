# SemiTPOT
Semi-Supervised Twitter PoS Tagging: 11-785: Midterm Project Report

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

If it doesn't work just install what you need

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install numpy scipy plotly seaborn pandas matplotlib
pip3 install conllu
pip3 install transformers
```

(Maybe you need to install torch separately)

3) Run the code

## Download from colab

```bash
python3 training.py
```
