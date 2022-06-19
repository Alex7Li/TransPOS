from glob import glob
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
import torch.nn.functional as F
from pathlib import Path
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Optional, Tuple
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
import torch.optim.lr_scheduler
import mapping_baselines

device = "cuda" if torch.cuda.is_available() else "cpu"

class MapperTrainingParameters:
    def __init__(self) -> None:
        self.batch_size = 16
        self.alpha = 0.0
        self.y_supervisor = None
        self.z_supervisor = None

    def use_supervision(self, new_alpha, y_dataset_name='tweebank', z_dataset_name='ark', model_name='vinai/bertweet-large'):
        self.alpha = new_alpha
        print("Getting Y supervisor, will train if no weights are cached")
        y_acc, self.y_supervisor = mapping_baselines.normal_model_baseline(y_dataset_name, model_name)
        print(f"Y supervisor accuracy {y_acc}")
        print("Getting Z supervisor, will train if no weights are cached")
        z_acc, self.z_supervisor = mapping_baselines.normal_model_baseline(z_dataset_name, model_name)
        print(f"Z supervisor accuracy {z_acc}")


def compose_loss(batch, model: MapperModel, globals: MapperTrainingParameters, input_label="y"):
    """
    Compute KL(D_z(E(x)), D_y(E(x),y), y) as described in
    the paper.

    (If input_label is z, compute the other term, but the
    variable names will assume the first term)
    """
    decode_y = model.decode_y if input_label == "y" else model.decode_z
    decode_z = model.decode_z if input_label == "y" else model.decode_y
    supervisor = globals.y_supervisor if input_label == "y" else globals.z_supervisor

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
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_f(y_pred_soft.flatten(0, 1), labels.flatten())
    pseudolabel_loss = torch.tensor(0)
    if supervisor is not None:
        super_logits = supervisor(**batch).logits
        pseudolabel_loss += F.kl_div(y_pred, super_logits)
    return loss, pseudolabel_loss, correct, total


def train_epoch(
    y_dataloader: TransformerCompatDataset,
    z_dataloader: TransformerCompatDataset,
    model: MapperModel,
    optimizer: torch.optim.Optimizer,
    globals: MapperTrainingParameters
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
        ly, ply, cy, ty = compose_loss(batch_y, model, globals, "y")
        lz, plz, cz, tz = compose_loss(batch_z, model, globals, "z")
        cross_entropy_loss = ly + lz
        psuedolabel_loss = globals.alpha * (ply + plz)
        total_loss = cross_entropy_loss + psuedolabel_loss
        postfix_dict = {'CE': cross_entropy_loss.item()}
        if globals.alpha != 0:
            postfix_dict.update({'psuedo': psuedolabel_loss.item()/globals.alpha})
        pbar.set_postfix(postfix_dict)
        total_loss.backward()
        correct_y += cy
        total_y += ty
        correct_z += cz
        total_z += tz
        optimizer.step()
        optimizer.zero_grad()
        sum_kl_loss += cross_entropy_loss.detach()
        sum_label_loss += psuedolabel_loss.detach()
    print(f"Train accuracy Y: {100 * correct_y / total_y:.3f}% Z: {100 * correct_z / total_z:.3f}%")
    return sum_kl_loss / n_iters, sum_label_loss / n_iters


def get_validation_predictions(model: MapperModel, shared_val_set, globals):
    model.eval()
    y_dataset, z_dataset = shared_val_set
    y_dataloader = training.get_dataloader(
        model.base_transformer_name, y_dataset, globals.batch_size, shuffle=False
    )
    z_dataloader = training.get_dataloader(
        model.base_transformer_name, z_dataset, globals.batch_size, shuffle=False
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
    (y_preds, y_labels), (z_preds, z_labels) = get_validation_predictions(model, shared_val_dataset, globals)
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
    globals: MapperTrainingParameters
):
    if load_weights and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded weights from {save_path}. Will continue for {n_epochs} more epochs")
    optimizer = torch.optim.NAdam([
        {'params': itertools.chain(model.model.parameters()),
            'lr': 3e-5, 'weight_decay': 1e-4},
        {'params': itertools.chain(model.yzdecoding.parameters(),
            model.zydecoding.parameters()),
         'lr': 3e-3, 'weight_decay': 1e-6},
        {'params': [model.soft_label_value], 'lr':1e-3,
         'weight_decay': 0},
        ])
    scheduler = LambdaLR(optimizer, lr_lambda=
        [
            lambda epoch:0 if epoch < 1 else 1,
            lambda epoch:max(1e-2,.5**epoch),
            lambda epoch:1
        ]
        )
    best_validation_acc = 0
    valid_acc = 0
    #if shared_val_dataset is not None:
    #     Test
    #    valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs",):
        kl_loss, label_loss = train_epoch(
            y_dataloader, z_dataloader, model, optimizer, globals
        )
        print(f"Epoch {epoch_index} Train CE Loss: {kl_loss} Train pseudolabel loss: {label_loss}")
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



def main(y_dataset_name, z_dataset_name, model_name, n_epochs=10, load_cached_weights=True, alpha=.01):
    save_path = Path("models") / (
        model_name.split("/")[-1] + "_mapper_" + y_dataset_name + "_" + z_dataset_name
    )
    y_dataset = training.get_dataset(y_dataset_name, "unshared")
    z_dataset = training.get_dataset(z_dataset_name, "unshared")
    globals = MapperTrainingParameters()
    globals.use_supervision(.1)
    y_dataloader = training.get_dataloader(
        model_name, y_dataset, globals.batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model_name, z_dataset, globals.batch_size, shuffle=True
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
        load_cached_weights, globals
    )
    # After 10 epochs:
    # Val Acc Y: 92.12633451957295% Val Acc Z 92.48220640569394%
    return mapped_model


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main("tweebank", "ark", "vinai/bertweet-large", load_cached_weights=False, alpha=0)
