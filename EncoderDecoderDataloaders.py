from functools import partial

import torch
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, leave=False, position=0, dynamic_ncols=True)
import warnings
from ArkDataset.load_ark import load_ark
from TPANNDataset.load_tpann import load_tpann
from TweeBankDataset.load_tweebank import load_tweebank


def main():
    tpann_train, tpann_val, tpann_test = load_tpann()
    # Needed Imports
    warnings.filterwarnings("ignore")


def get_shared_examples(ark_all, tweebank_all):
    """Takes as input all examples from ark and twee bank, then returns a list of shared examples of the format
    [shared_x,ark_label,twee_label]"""
    shared_examples = []
    for ark in tqdm(ark_all, desc="Finding shared examples"):
        for twee in tweebank_all:
            if ark[0] == twee[0]:
                shared_examples.append([])
                shared_examples[-1].extend([ark[0], ark[1], twee[1]])
    print("Number of shared Examples: ", len(shared_examples))
    return shared_examples


class ArkSharedDataset(torch.utils.data.Dataset):
    def __init__(self, shared_examples, test=False):
        self.shared_examples = shared_examples
        self.X = []
        self.Y = []
        for ex in shared_examples:
            self.X.append(ex[0])
            self.Y.append(ex[1])
        if test:
            self.X = self.X[-50:]
            self.Y = self.Y[-50:]
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        return x, y


class TweeSharedDataset(torch.utils.data.Dataset):
    def __init__(self, shared_examples, test=False):
        self.shared_examples = shared_examples
        self.X = []
        self.Y = []
        for ex in shared_examples:
            self.X.append(ex[0])
            self.Y.append(ex[2])
        if test:
            self.X = self.X[-50:]
            self.Y = self.Y[-50:]
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        return x, y


def create_tweebank_ark_dataset():
    ark_train, ark_val, ark_test = load_ark()
    tweebank_train, tweebank_val, tweebank_test = load_tweebank()
    ark_all = ark_train + ark_val + ark_test
    tweebank_all = tweebank_train + tweebank_val + tweebank_test
    shared_examples = get_shared_examples(ark_all, tweebank_all)
    ark_shared_dataset = ArkSharedDataset(shared_examples)
    twee_shared_dataset = TweeSharedDataset(shared_examples)
    assert len(ark_shared_dataset) == len(twee_shared_dataset)
    return twee_shared_dataset, ark_shared_dataset


if __name__ == "__main__":
    main()
