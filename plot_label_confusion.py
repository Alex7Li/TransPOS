import numpy as np
from dataloading_utils import get_dataset_mapping
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from training import pipeline, get_dataset, validation_epoch, load_model, get_dataloader
import torch
import os


def invert_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def plot_label_confusion(preds, labels, train_dataset_name, val_dataset_name):
    """
    Y-axis represents the labels of the validation dataset
    X-axis represents the labels of the train dataset
    The model predicts a label in the X-axis and this label is converted to the unified label
    This converted unified label is compared to the actual unified label from Tweebank.
    If they are different, then the cell is negative, otherwise the cell is positive.
    The Y-axis indicates what the unified label was supposed to be
    """
    n_label_classes = np.max(labels) + 1
    n_pred_classes = np.max(preds) + 1
    train_name_map, train_to_unified = get_dataset_mapping(train_dataset_name)
    val_name_map, val_to_unified = get_dataset_mapping(val_dataset_name)

    train_reorder = invert_permutation(
        sorted(range(n_pred_classes), key=lambda i: train_to_unified[train_name_map[i]])
    )
    val_reorder = invert_permutation(
        sorted(range(n_label_classes), key=lambda i: val_to_unified[val_name_map[i]])
    )
    confusion_matrix = np.zeros((n_label_classes, n_pred_classes))
    for l, p in zip(labels, preds):
        confusion_matrix[val_reorder[l]][train_reorder[p]] += 1
    # Negate incorrect labels so that we can actually read the thing
    for l in range(n_label_classes):
        for p in range(n_pred_classes):
            if val_to_unified[val_name_map[l]] != train_to_unified[train_name_map[p]]:
                confusion_matrix[val_reorder[l]][train_reorder[p]] *= -1
    if np.sum(confusion_matrix) < 0:
        print("Uhh the accuracy is terrible, you probs have a bug")
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=[
            val_name_map[invert_permutation(val_reorder)[l]]
            for l in range(n_label_classes)
        ],
        columns=[
            train_name_map[invert_permutation(train_reorder)[p]]
            for p in range(n_pred_classes)
        ],
    )
    plt.figure(figsize=(n_pred_classes * 3 // 4, n_label_classes * 3 // 4))
    heatmap = seaborn.heatmap(df_cm, annot=True, fmt="g")
    heatmap.set_xlabel(train_dataset_name)
    heatmap.set_ylabel(val_dataset_name)
    plt.savefig(
        os.path.join(
            "Results",
            f"label_confusion_train_{train_dataset_name}_val_{val_dataset_name}.png",
        )
    )


def get_model_predictions_and_true_labels(hparams, val_dataset_name):
    hparams["save_path"] = os.path.join(
        "models",
        "teacher_" + hparams["model_name"].split("/")[-1] + "_" + hparams["dataset"],
    )
    if os.path.exists(hparams["save_path"]):
        dataset = get_dataset(hparams["dataset"], "train")
        model = load_model(hparams["model_name"], dataset.num_labels)
        model.load_state_dict(torch.load(hparams["save_path"]))
    else:
        model = pipeline(hparams)

    val_dataset = get_dataset(val_dataset_name, "val")
    val_dataloader = get_dataloader(
        hparams["model_name"], val_dataset, hparams["batch_size"]
    )
    preds, labels = validation_epoch(model, val_dataloader)

    return preds, labels


if __name__ == "__main__":
    preds_tpann, labels_tpann = get_model_predictions_and_true_labels(
        {
            "n_epochs": 10,
            "batch_size": 8,
            "dataset": "ark",
            "model_name": "vinai/bertweet-large",
        },
        "tweebank",
    )

    plot_label_confusion(preds_tpann, labels_tpann, "ark", "GUM")
