from glob import glob
from augmented_datasets import get_dataloader
from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset, flatten_preds_and_labels
import torch
from collections import defaultdict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, leave=True, position=0, dynamic_ncols=True)
import training
import itertools
import torch.nn.functional as F
from pathlib import Path
import math
import os
import gc
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Optional, Tuple
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
import torch.optim.lr_scheduler
from transformers import get_scheduler


class MapperTrainingParameters:
    batch_size = 16
    def __init__(self) -> None:
        self.alpha = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def use_supervision(self, new_alpha):
        self.alpha = new_alpha


def compose_loss(batch, model: MapperModel, globals: MapperTrainingParameters, input_label="y"):
    """
    Compute KL(D_z(E(x)), D_y(E(x),y), y) as described in
    the paper.

    (If input_label is z, compute the other term, but the
    variable names will assume the first term)
    """
    # decode_y = model.decode_y if input_label == "y" else model.decode_z
    decode_z = model.decode_z if input_label == "y" else model.decode_y
    supervisor_y = model.ydecoding if input_label == "y" else model.zdecoding
    supervisor_z = model.zdecoding if input_label == "y" else model.ydecoding

    batch = {k: v.to(globals.device) for k, v in batch.items()}
    labels = batch["labels"]
    del batch["labels"]
    e_y = model.encode(batch)
    # z_tilde = decode_y(e_y, labels)
    z_tilde = supervisor_z(e_y)
    y_tilde = decode_z(e_y, z_tilde)
    y_pred_probs = F.softmax(y_tilde, dim=2)
    y_pred = torch.argmax(y_pred_probs, dim=2)
    correct = torch.sum(y_pred == labels)
    total = torch.sum(labels != -100)
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-100)
    losses = {}
    losses['full CE'] = loss_f(y_pred_probs.flatten(0, 1), labels.flatten())
    if globals.alpha is not None:
        # supervisor should be accurate
        super_y_logits = supervisor_y(e_y)
        super_probs = F.softmax(super_y_logits, dim=2)
        losses['supervised CE'] = loss_f(super_probs.flatten(0, 1),
            labels.flatten()) * globals.alpha
    return losses, correct, total


def train_epoch(
    y_dataloader: TransformerCompatDataset,
    z_dataloader: TransformerCompatDataset,
    model: MapperModel,
    optimizer: torch.optim.Optimizer,
    globals: MapperTrainingParameters
):
    model.train()
    n_iters = min(len(y_dataloader), len(z_dataloader))
    avg_losses = defaultdict(lambda:0.0)
    correct_y = 0
    correct_z = 0
    total_y = 0
    total_z = 0
    pbar = tqdm(zip(y_dataloader, z_dataloader), total=n_iters)
    # Iterate until either dataset is exhausted.
    for batch_y, batch_z in pbar:
        batch_loss_y, cy, ty = compose_loss(batch_y, model, globals, "y")
        batch_loss_z, cz, tz = compose_loss(batch_z, model, globals, "z")
        losses = batch_loss_y
        total_loss = torch.tensor(0.0)
        for k, v in batch_loss_z.items():
            losses[k] += v
            total_loss += v.item()
            if k in avg_losses:
                avg_losses[k] = v.item()
            else:
                avg_losses[k] = avg_losses[k] * .95 + v.item() *.05
        pbar.set_postfix({k: v.item() for k,v in losses.items()})
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct_y += cy
        total_y += ty
        correct_z += cz
        total_z += tz

    print(f"Train accuracy Y: {100 * correct_y / total_y:.3f}% Z: {100 * correct_z / total_z:.3f}%")


def get_validation_predictions(model: MapperModel, shared_val_set):
    model.eval()
    y_dataset, z_dataset = shared_val_set
    y_dataloader = training.get_dataloader(
        model.base_transformer_name, y_dataset, MapperTrainingParameters.batch_size, shuffle=False
    )
    z_dataloader = training.get_dataloader(
        model.base_transformer_name, z_dataset, MapperTrainingParameters.batch_size, shuffle=False
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
    load_weights,
    parameters: MapperTrainingParameters
):
    if load_weights and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded weights from {save_path}. Will continue for {n_epochs} more epochs")
    optimizer = torch.optim.NAdam([
        {'params': model.model.parameters(),
            'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': itertools.chain(
            model.yzdecoding.parameters(),
            model.zydecoding.parameters(),
            model.ydecoding.parameters(),
            model.zdecoding.parameters()),
         'lr': 2e-5, 'weight_decay': 1e-6},
        {'params': [model.soft_label_value], 'lr':1e-3,
         'weight_decay': 0},
        ])
    # scheduler = LambdaLR(optimizer, lr_lambda=
    #     [ lambda epoch:1, lambda epoch:1, lambda epoch:1 ]
    #     )
    scheduler: torch.optim.lr_scheduler.LinearLR = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=min(len(y_dataloader), len(z_dataloader)) * n_epochs,
    ) # type:ignore
    best_validation_acc = 0
    valid_acc = 0
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs",):
        train_epoch(
            y_dataloader, z_dataloader, model, optimizer, parameters
        )
        if shared_val_dataset is not None:
            valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
            valid_acc = math.sqrt(valid_acc_y * valid_acc_z) # Geometric Mean
            print(f"Val Acc Y: {valid_acc_y*100:.3f}% Val Acc Z {valid_acc_z*100:.3f}%  Soft Label: {model.soft_label_value:.5f}")
        if valid_acc >= best_validation_acc:
            best_validation_acc = valid_acc
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(model.state_dict(), save_path)
        scheduler.step()
    return model



def main(y_dataset_name, z_dataset_name, model_name, n_epochs=10,
         load_cached_weights=True, parameters=None):
    if parameters == None:
        parameters = MapperTrainingParameters()
        parameters.use_supervision(1.0)
    save_path = Path("models") / (
        model_name.split("/")[-1] + "_mapper_" + y_dataset_name + "_" + z_dataset_name
    )
    y_dataset = training.get_dataset(y_dataset_name, "unshared")
    z_dataset = training.get_dataset(z_dataset_name, "unshared")
    y_dataloader = training.get_dataloader(
        model_name, y_dataset, MapperTrainingParameters.batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model_name, z_dataset, MapperTrainingParameters.batch_size, shuffle=True
    )
    model = MapperModel(
        "vinai/bertweet-large", y_dataset.num_labels, z_dataset.num_labels
    )
    model.to(parameters.device)
    shared_val_dataset = None
    if y_dataset_name == "tweebank" and z_dataset_name == "ark":
        shared_val_dataset = create_tweebank_ark_dataset()
    mapped_model = train_model(
        model, y_dataloader, z_dataloader, shared_val_dataset, n_epochs, save_path,
        load_cached_weights, parameters
    )
    # After 10 epochs:
    # Val Acc Y: 92.12633451957295% Val Acc Z 92.48220640569394%
    return mapped_model


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large",
         load_cached_weights=False,
         n_epochs=20)
