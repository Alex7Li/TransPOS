from augmented_datasets import get_dataloader
from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset, flatten_preds_and_labels
import torch
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, leave=True, position=0, dynamic_ncols=True)
import training
import itertools
from typing import Tuple
import torch.nn.functional as F
from pathlib import Path
import math
import os
from typing import List
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
import torch.optim.lr_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16


def compose_loss(batch, model: MapperModel, input_label="y"):
    """
    Compute KL(D_z(E(x)), D_y(E(x),y), y) as described in
    the paper.

    (If input_label is z, compute the other term, but the
    variable names will assume the first term)
    """
    decode_y = model.decode_y if input_label == "y" else model.decode_z
    decode_z = model.decode_z if input_label == "y" else model.decode_y

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"]
    del batch["labels"]
    e_y = model.encode(batch)
    z_tilde = decode_y(e_y, labels)
    y_tilde = decode_z(e_y, z_tilde)
    y_pred_soft = F.softmax(y_tilde, dim=2)
    y_pred = torch.argmax(y_pred_soft, dim=2)
    correct = torch.sum(y_pred == labels)
    total = torch.sum(labels != -100)
    loss_f = torch.nn.CrossEntropyLoss()
    loss = loss_f(y_pred_soft.flatten(0, 1), labels.flatten())
    label_loss = model.label_loss(z_tilde, batch['attention_mask'])
    return loss, label_loss, correct, total


def train_epoch(
    y_dataloader: TransformerCompatDataset,
    z_dataloader: TransformerCompatDataset,
    model: MapperModel,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    n_iters = min(len(y_dataloader), len(z_dataloader))
    sum_kl_loss = 0
    sum_label_loss = 0
    correct_y = 0
    correct_z = 0
    total_y = 0
    total_z = 0
    pbar = tqdm(zip(y_dataloader, z_dataloader), total=n_iters)
    # Iterate until either dataset is exhausted.
    for batch_y, batch_z in pbar:
        ly, lly, cy, ty = compose_loss(batch_y, model, "y")
        lz, llz, cz, tz = compose_loss(batch_z, model, "z")
        loss = ly + lz
        label_loss = lly + llz
        total_loss = loss + label_loss
        pbar.set_description(f"CE:{loss:.2f}, soft_label:{label_loss:.2f}")
        total_loss.backward()
        correct_y += cy
        total_y += ty
        correct_z += cz
        total_z += tz
        optimizer.step()
        optimizer.zero_grad()
        sum_kl_loss += loss.detach()
        sum_label_loss += loss.detach()
    print(f"Train accuracy Y: {100 * correct_y / total_y:.3f}% Z: {100 * correct_z / total_z:.3f}%")
    return sum_kl_loss / n_iters, sum_label_loss / n_iters


def get_validation_predictions(model: MapperModel, shared_val_set):
    model.eval()
    y_dataset, z_dataset = shared_val_set
    y_dataloader = training.get_dataloader(
        model.base_transformer_name, y_dataset, batch_size, shuffle=False
    )
    z_dataloader = training.get_dataloader(
        model.base_transformer_name, z_dataset, batch_size, shuffle=False
    )
    predicted_y = []
    predicted_z = []
    labels_y = []
    labels_z = []
    for y_batch, z_batch in tqdm(
        zip(y_dataloader, z_dataloader),
        desc="Predicting Validation labels",
        total=len(y_dataloader),
    ):
        labels_y.append(y_batch['labels'])
        labels_z.append(z_batch['labels'])
        e = model.encode(y_batch)
        z_pred = torch.argmax(model.decode_y(e, labels_y[-1]), dim=2)
        y_pred = torch.argmax(model.decode_z(e, labels_z[-1]), dim=2)
        predicted_z.append(z_pred)
        predicted_y.append(y_pred)
    return flatten_preds_and_labels(predicted_y, labels_y), flatten_preds_and_labels(predicted_z, labels_z)

def model_validation_acc(model: MapperModel, shared_val_dataset) -> Tuple[float, float]:
    (y_preds, y_labels), (z_preds, z_labels) = get_validation_predictions(model, shared_val_dataset)
    y_acc = dataloading_utils.get_acc(y_preds, y_labels)
    z_acc = dataloading_utils.get_acc(z_preds, z_labels)
    return y_acc, z_acc


def train_model(
    model: MapperModel,
    y_dataloader,
    z_dataloader,
    shared_val_dataset,
    n_epochs,
    save_path,
    load_weights
):
    if load_weights and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded weights from {save_path}. Will continue for {n_epochs} more epochs")
    optimizer = torch.optim.NAdam([
        {'params': model.model.parameters(), 'lr': 3e-5, 'weight_decay': 1e-4},
        {'params': itertools.chain(model.yzdecoding.parameters(),
                                   model.zydecoding.parameters()),
         'lr': 1e-4, 'weight_decay': 6e-5},
        {'params': [model.soft_label_value], 'lr': 1e-3, 'weight_decay': 0},
        ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, patience=8, verbose=True)
    best_validation_acc = 0
    valid_acc = 0
    #if shared_val_dataset is not None:
    #     Test
    #    valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs",):
        kl_loss, label_loss = train_epoch(
            y_dataloader, z_dataloader, model, optimizer
        )
        print(f"Epoch {epoch_index} Train KL Loss: {kl_loss} Train label loss: {label_loss}")
        if shared_val_dataset is not None:
            valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
            valid_acc = math.sqrt(valid_acc_y * valid_acc_z) # Geometric Mean
            print(f"Val Acc Y: {valid_acc_y*100:.3f}% Val Acc Z {valid_acc_z*100:.3f}% Soft label {model.soft_label_value:.5f}")
        if valid_acc >= best_validation_acc:
            best_validation_acc = valid_acc
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(model.state_dict(), save_path)
        scheduler.step(valid_acc)
    return model



def main(y_dataset_name, z_dataset_name, model_name, n_epochs=10, load_cached_weights=True):
    save_path = Path("models") / (
        model_name.split("/")[-1] + "_mapper_" + y_dataset_name + "_" + z_dataset_name
    )
    y_dataset = training.get_dataset(y_dataset_name, "unshared")
    z_dataset = training.get_dataset(z_dataset_name, "unshared")
    y_dataloader = training.get_dataloader(
        model_name, y_dataset, batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model_name, z_dataset, batch_size, shuffle=True
    )
    model = MapperModel(
        "vinai/bertweet-large", y_dataset.num_labels, z_dataset.num_labels
    )
    model.to(device)
    shared_val_dataset = None
    if y_dataset_name == "tweebank" and z_dataset_name == "ark":
        shared_val_dataset = create_tweebank_ark_dataset()
    mapped_model = train_model(
        model, y_dataloader, z_dataloader, shared_val_dataset, n_epochs, save_path,
        load_weights=load_cached_weights
    )
    # After 10 epochs:
    # Val Acc Y: 92.12633451957295% Val Acc Z 92.48220640569394%
    return mapped_model


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large", load_cached_weights=False)
