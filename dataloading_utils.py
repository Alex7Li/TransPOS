from typing import *
import torch
import torch.utils.data
from TweeBankDataset.load_tweebank import TWEEBANK_POS_MAPPING

PYTORCH_IGNORE_INDEX = (
    -100
)  # Index ignored by pytorch https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html


def create_pos_mapping(pos_tags: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates two dictionaries
    - The pos tags are mapped to their indices in the list and vice versa
    ----------------------------------------------------------------------
    Args
    -----
    ark_pos_tags: List[str]
      The list of pos tags used in the ark dataset

    Returns
    --------
    ark_pos_index_mapping: Dict[str, int]
      The mapping of pos to their indices

    ark_index_pos_mapping: Dict[int, str]
      The mapping of the indices to their corresponding pos
    """
    index_mapping = {tag: i for i, tag in enumerate(pos_tags)}
    index_pos_mapping = {v: k for k, v in index_mapping.items()}
    return index_mapping, index_pos_mapping


# https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
def tokenize_and_align_labels(orig_x, orig_labels, tokenizer):
    """
    Tokenize x, and lengthen the original labels similarly.
    It handles start/end special tokens as well as cases where a word is
    split into many words.
    orig_x: A list of words
    orig_labels_example: A list of label indicies
    output: data formatted the way we want
    """
    tokenized_inputs = tokenizer(orig_x, truncation=True, is_split_into_words=True)

    word_ids = tokenized_inputs.word_ids()
    label_ids = []
    for i, word_idx in enumerate(word_ids):

        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            label_ids.append(PYTORCH_IGNORE_INDEX)
            continue
        if i + 1 >= len(word_ids):
            continue  # Just in case the EOS is cut off or something
        next_word_idx = word_ids[i + 1]
        # We set the label for the last token of each word.
        # Not the first, because gpt2 is unidirectional so it
        # needs to read em all
        if next_word_idx == word_idx:
            label_ids.append(PYTORCH_IGNORE_INDEX)
        else:
            label_ids.append(orig_labels[word_idx])
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


class TransformerCompatDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset: torch.utils.data.Dataset, tokenizer):
        self.orig_dataset = orig_dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        words, labels = self.orig_dataset[idx]
        encoding = tokenize_and_align_labels(words, labels, self.tokenizer)
        return encoding

    def __len__(self):
        return len(self.orig_dataset)

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.L1 = len(self.dataset1)
        self.L2 = len(self.dataset2)
        self.L3 = 0 if dataset3 == None else len(self.dataset3)
        self.total_len = self.L1 + self.L2 + self.L3
        self.num_labels = None
        if self.dataset1.num_labels == self.dataset2.num_labels:
            if (
                self.dataset3 is None
                or self.dataset1.num_labels == self.dataset3.num_labels
            ):
                self.num_labels = self.dataset1.num_labels

    def __len__(self):
        return self.total_len

    def __getitem__(self, ind):
        if ind < self.L1:
            return self.dataset1[ind]
        elif ind < self.L1 + self.L2:
            return self.dataset2[ind - self.L1]
        else:
            return self.dataset3[ind - self.L1 - self.L2]


def get_num_examples(dataloader):
    total_examples = 0
    for batch in dataloader:
        total_examples += len(batch)
    return total_examples


def map_index_to_tag(indices, mapping):
    """
    Map an index to the appropriate pos tag using the mapping for a dataset
    ------------------------------------------------------------------------
    Args
    -----
    1. indices: List[int]
      List of integers where each int corresponds to a pos tag
    2. mapping: Dict[int, str]
      Dataset specific mapping of int to pos tag

    Returns
    --------
    The actual pos tags
    """

    return [mapping[index] for index in indices]


def map_labels_to_twee(pos_tags, mapping):
    """
    Map the pos tags of a dataset to the unified pos tags
    ------------------------------------------------------
    Args
    -----
    1. pos_tags: List[str]
      List of pos tags from a particular dataset
    2. mapping: Dict[str, str]
      Maps the pos tags from a dataset to the unified pos tags

    Returns
    --------
    Unified pos tags for the dataset
    """
    return [mapping[tag] for tag in pos_tags]


def filter_negative_hundred(preds, labels):
    """
    Flatten a list of tensors
    --------------------------
    Args
    -----
    1. preds: List[Tensor]
      A list of 2d tensor preds
    2. labels: List[Tensor]
      A list of 2d tensor label

    Returns
    --------
    Two lists (pred, label)
    """
    flattened_preds = [tensor.flatten() for tensor in preds]
    final_preds = torch.cat(flattened_preds, dim=0)
    final_preds = [tensor.item() for tensor in final_preds]

    flattened_labels = [tensor.flatten() for tensor in labels]
    final_labels = torch.cat(flattened_labels, dim=0)
    final_labels = [tensor.item() for tensor in final_labels]
    assert len(final_preds) == len(final_labels)

    pred_labels = [pair for pair in zip(final_preds, final_labels) if pair[1] != -100]
    final_preds = [pair[0] for pair in pred_labels]
    final_labels = [pair[1] for pair in pred_labels]
    return final_preds, final_labels


def get_dataset_mapping(dataset_name) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Get the index to pos mapping for the original dataset
    Get the dataset pos to unified pos mapping
    ------------------------------------------------------
    Args
    -----
    1. dataset_name: str
      The dataset that the model making predictions has been trained on

    Returns
    --------
    1. index_pos_mapping: Dict[int, str]
      Maps the integer predictions to their pos tag for dataset_name
    2. dataset_to_twee_pos_mapping: Dict[str, str]
      Maps the pos tags for dataset_name to unified pos tags
    """
    if dataset_name == "ark":
        ARK_POS_TAGS = [
            "N",
            "O",
            "^",
            "S",
            "Z",
            "V",
            "A",
            "R",
            "!",
            "D",
            "P",
            "&",
            "T",
            "X",
            "#",
            "@",
            "~",
            "U",
            "E",
            "$",
            ",",
            "G",
            "L",
            "M",
            "Y",
        ]
        pos_index_mapping, index_pos_mapping = create_pos_mapping(ARK_POS_TAGS)
        dataset_to_twee_pos_mapping = {
            "N": "NOUN",
            "O": "PRON",
            "^": "PROPN",
            "S": "PRON",
            "Z": "PROPN",
            "V": "VERB",
            "A": "ADJ",
            "R": "ADV",
            "!": "INTJ",
            "D": "DET",
            "P": "ADP",
            "&": "CCONJ",
            "T": "ADP",
            "X": "DET",
            "#": "X",
            "@": "X",
            "~": "PUNCT",
            "U": "X",
            "E": "SYM",
            "$": "NUM",
            ",": "PUNCT",
            "G": "X",
            "L": "AUX",
            "M": "PROPN",
            "Y": "DET",
        }
        assert set(ARK_POS_TAGS) == set(dataset_to_twee_pos_mapping)
    elif dataset_name == "atis" or dataset_name == "GUM" or dataset_name == "tweebank":
        # all_pos is run from above as a global
        pos_index_mapping = TWEEBANK_POS_MAPPING
        index_pos_mapping = {v: k for k, v in pos_index_mapping.items()}
        dataset_to_twee_pos_mapping = {v: v for v in index_pos_mapping.values()}
        assert set(TWEEBANK_POS_MAPPING.keys()) == set(dataset_to_twee_pos_mapping)
    elif dataset_name == "TPANN":
        TPANN_POS_TAGS = [
            ":",
            "VBZ",
            "PRP$",
            "WRB",
            "MD",
            "RB",
            "NNS",
            "DT",
            "UH",
            "VBG",
            "]",
            "NN",
            "URL",
            "VBD",
            ".",
            "VBP",
            "POS",
            "WP",
            "RT",
            "VB",
            "HT",
            ")",
            "VBN",
            "PRP",
            "TO",
            "NNP",
            "JJR",
            "USR",
            "RP",
            "SYM",
            ",",
            "JJ",
            "O",
            "CC",
            "''",
            "CD",
            "(",
            "PDT",
            "IN",
            "[",
            "WDT",
            "JJS",
            "RBR",
            "NNPS",
            "LS",
            "RBS",
            "FW",
            "EX",
            "WP$",
        ]
        dataset_to_twee_pos_mapping = {
            "CD": "NUM",
            "LS": "NUM",
            "NNP": "PROPN",
            "NNPS": "PROPN",
            "SYM": "SYM",
            "JJ": "ADJ",
            "JJR": "ADJ",
            "JJS": "ADJ",
            "RB": "ADV",
            "WRB": "ADV",
            "RBR": "ADV",
            "RBS": "ADV",
            "DT": "DET",
            "PDT": "DET",
            "EX": "DET",
            "WDT": "PRON",
            "FW": "X",
            "URL": "X",
            "USR": "X",
            "HT": "X",
            "RT": "X",
            "POS": "PART",
            "TO": "PART",
            "RP": "ADP",
            "UH": "INTJ",
            "PRP": "PRON",
            "WP": "PRON",
            "WP$": "PRON",
            "CC": "CCONJ",
            "IN": "ADP",
            "NN": "NOUN",
            "NNS": "NOUN",
            "O": "PUNCT",
            "MD": "AUX",
            "VB": "VERB",
            "VBD": "VERB",
            "VBG": "VERB",
            "VBN": "VERB",
            "VBP": "VERB",
            "VBZ": "VERB",
            ":": "PUNCT",
            "]": "PUNCT",
            "''": "PUNCT",
            "(": "PUNCT",
            "[": "PUNCT",
            ")": "PUNCT",
            ",": "PUNCT",
            "PRP$": "PRON",
            ".": "PUNCT",
        }

        assert set(TPANN_POS_TAGS) == set(dataset_to_twee_pos_mapping)
        pos_index_mapping, index_pos_mapping = create_pos_mapping(TPANN_POS_TAGS)
    else:
        index_pos_mapping = None
        dataset_to_twee_pos_mapping = None
        print("no dataset")

    return index_pos_mapping, dataset_to_twee_pos_mapping


def get_acc(preds, labels):
    """
    Get the accuracy of unified predictions
    ----------------------------------------
    Args
    -----
    1. preds: List[str]
      Model predictions
    2. labels: List[str]
      The actual labels of the dataset

    Returns
    --------
    Accuracy of predictions on the dataset
    """
    num_correct = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            num_correct += 1
    if len(preds) == 0:
        return 0
    else:
        return num_correct / len(preds)


def get_validation_acc(preds, labels, train_dataset_name, val_dataset_name):
    train_index_pos_mapping, train_dataset_to_twee_pos_mapping = get_dataset_mapping(
        train_dataset_name
    )

    pos_preds = map_index_to_tag(preds, train_index_pos_mapping)
    unified_pos_preds = map_labels_to_twee(pos_preds, train_dataset_to_twee_pos_mapping)

    val_index_pos_mapping, val_dataset_to_twee_pos_mapping = get_dataset_mapping(
        val_dataset_name
    )
    pos_true = map_index_to_tag(labels, val_index_pos_mapping)
    unified_pos_labels = map_labels_to_twee(pos_true, val_dataset_to_twee_pos_mapping)

    return get_acc(unified_pos_preds, unified_pos_labels)
