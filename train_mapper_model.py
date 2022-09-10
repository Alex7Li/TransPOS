from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset, flatten_preds_and_labels
import torch
import torch.optim
from collections import defaultdict
from functools import partial
from tqdm import tqdm as std_tqdm
import numpy as np
tqdm = partial(std_tqdm, leave=True, position=0, dynamic_ncols=True)
import training
import itertools
import torch.nn.functional as F
from pathlib import Path
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Optional
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
from transformers import get_scheduler

class MapperTrainingParameters:
    def __init__(
        self,
        total_epochs=20,
        only_supervised_epochs=0, # Can increase for a comparison
        alpha: Optional[float] = 1.0,
        batch_size=16,
        tqdm=False,
        x_dropout=.1,
        lr=3e-4,
        lr_fine_tune=3e-5,
        lr_warmup_epochs=0,
        use_separate_encoder=False
    ) -> None:
        super()
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert 0 <= only_supervised_epochs <= total_epochs
        self.total_epochs =  total_epochs
        self.only_supervised_epochs = only_supervised_epochs
        self.batch_size = batch_size
        self.tqdm=tqdm
        self.x_dropout=x_dropout
        self.lr=lr
        self.lr_fine_tune=lr_fine_tune
        # Separate encoder with some shared weights 
        # https://arxiv.org/abs/0907.1815
        # Very slow since you need to recompute encoder twice.
        self.use_separate_encoder=use_separate_encoder
        self.lr_warmup_epochs=lr_warmup_epochs
        if self.alpha == None:
            assert only_supervised_epochs == 0


def compose_loss(
    batch,
    model: MapperModel,
    parameters: MapperTrainingParameters,
    input_label: str,
    epoch_ind: int,
):
    """
    Compute KL(D_z(E(x)), D_y(E(x),y), y) as described in
    the paper.

    (If input_label is z, compute the other term, but the
    variable names will assume the first term)
    """
    decode_y = model.decode_y if input_label == "y" else model.decode_z
    encode_y = model.encode_y  if input_label == "y" else model.encode_z
    encode_z = model.encode_z  if input_label == "y" else model.encode_y
    supervisor_y = model.ydecoding if input_label == "y" else model.zdecoding
    supervisor_z = model.zdecoding if input_label == "y" else model.ydecoding

    batch = {k: v.to(parameters.device) for k, v in batch.items()}
    labels = batch['labels']
    e_y = encode_y(batch)
    losses = {}
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-100)
    if epoch_ind >= parameters.only_supervised_epochs:
        if parameters.use_separate_encoder:
            e_z = encode_z(batch)
            z_tilde = supervisor_z(e_z)
            y_tilde = decode_y(e_z, z_tilde)
        else:
            z_tilde = supervisor_z(e_y)
            y_tilde = decode_y(e_y, z_tilde)
        losses["full CE"] = loss_f(y_tilde.flatten(0, 1), labels.flatten())
        y_pred = torch.argmax(y_tilde, dim=2)
        correct = torch.sum(y_pred == labels)
        total = torch.sum(labels != -100)
    if parameters.alpha is not None:
        super_y_logits = supervisor_y(e_y)
        ce_loss = (
            loss_f(super_y_logits.flatten(0, 1), labels.flatten()) * parameters.alpha
        )
        losses["supervised CE"] = ce_loss
        y_pred = torch.argmax(super_y_logits, dim=2)
        correct = torch.sum(y_pred == labels)
        total = torch.sum(labels != -100)
    return losses, correct, total


def train_epoch(
    y_dataloader: TransformerCompatDataset,
    z_dataloader: TransformerCompatDataset,
    model: MapperModel,
    optimizer: torch.optim.Optimizer,
    parameters: MapperTrainingParameters,
    cur_epoch: int,
):
    model.train()
    avg_losses = defaultdict(lambda: 0.0)
    correct_y = 0
    correct_z = 0.0001
    total_y = 0
    total_z = 0.0001
    pbar = zip(y_dataloader, z_dataloader)
    if parameters.tqdm:
        n_iters = min(len(y_dataloader), len(z_dataloader))
        pbar = tqdm(pbar, total=n_iters)
    # Iterate until either dataset is exhausted.
    for batch_y, batch_z in pbar:
        batch_loss_y, cy, ty = compose_loss(batch_y, model, parameters, "y", cur_epoch)
        batch_loss_z, cz, tz = compose_loss(batch_z, model, parameters, "z", cur_epoch)
        losses = batch_loss_y
        total_loss = torch.tensor(0.0).to(parameters.device)
        for k, v in batch_loss_z.items():
            losses[k] += v
        for k, batch_loss in losses.items():
            total_loss += batch_loss
            if k in avg_losses:
                avg_losses[k] = batch_loss.item()
            else:
                avg_losses[k] = avg_losses[k] * 0.95 + batch_loss.item() * 0.05
        if parameters.tqdm:
            pbar.set_postfix({k: v for k, v in avg_losses.items()})
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct_y += cy
        total_y += ty
        correct_z += cz
        total_z += tz
    if cur_epoch >= parameters.only_supervised_epochs:
        print("no label input train accuracy", end=" ")
    else:
        print("supervised train accuracy", end=" ")
    print(
        f"Y: {100 * correct_y / total_y:.3f}% Z: {100 * correct_z / total_z:.3f}%"
    )
    print(f"Train losses: {avg_losses.items()}")


def get_validation_predictions(
    model: MapperModel, shared_val_set, inference_type: str, parameters: MapperTrainingParameters
):
    model.eval()
    y_dataset, z_dataset = shared_val_set
    y_dataloader = training.get_dataloader(
        model.base_transformer_name,
        y_dataset,
        parameters.batch_size,
        shuffle=False,
    )
    z_dataloader = training.get_dataloader(
        model.base_transformer_name,
        z_dataset,
        parameters.batch_size,
        shuffle=False,
    )
    predicted_y = []
    predicted_z = []
    labels_y = []
    labels_z = []
    pbar = zip(y_dataloader, z_dataloader)
    if parameters.tqdm:
        pbar = tqdm(pbar, 
        desc="Predicting Validation labels",
        total=len(y_dataloader),
        )
    for y_batch, z_batch in pbar:
        y_true =  y_batch['labels']
        z_true =  z_batch['labels']
        e_y = model.encode_y(y_batch)
        if parameters.use_separate_encoder:
            e_z = model.encode_z(z_batch)
        else:
            e_z = e_y # Don't re-run the encoder, will be slow
        if inference_type == "ours":
            z_pred = torch.argmax(model.decode_z(e_z, y_true), dim=2)
            y_pred = torch.argmax(model.decode_y(e_y, z_true), dim=2)
        elif inference_type == "x baseline":
            y_pred = torch.argmax(model.ydecoding(e_y), dim=2)
            z_pred = torch.argmax(model.zdecoding(e_z), dim=2)
        elif inference_type == "no_label_input":
            y_pred = torch.argmax(model.decode_y(e_y, model.zdecoding(e_z)), dim=2)
            z_pred = torch.argmax(model.decode_z(e_z, model.ydecoding(e_y)), dim=2)
        elif inference_type == "independent":
            y_pred = torch.argmax(model.ydecoding(e_y) + model.decode_y(e_y, z_true), dim=2)
            z_pred = torch.argmax(model.zdecoding(e_z) + model.decode_z(e_z, y_true), dim=2)
        else:
            raise NotImplementedError()
        labels_y.append(y_true)
        labels_z.append(z_true)
        predicted_z.append(z_pred)
        predicted_y.append(y_pred)
    return flatten_preds_and_labels(predicted_y, labels_y), flatten_preds_and_labels(
        predicted_z, labels_z
    )


def model_validation_acc(
    model: MapperModel,
    shared_val_dataset,
    cur_epoch: int,
    parameters: MapperTrainingParameters,
    do_others: bool
) -> Tuple[float, float]:
    model.eval()
    if cur_epoch < parameters.only_supervised_epochs:
        if do_others:
            val_types = ["no_label_input", "ours"]
        else:
            val_types = ["x baseline"]
    else:
        if do_others:
            val_types = ["no_label_input", "x baseline"]
        else:
            val_types = ["ours"]
    for val_type in val_types:
        (y_preds, y_labels), (z_preds, z_labels) = get_validation_predictions(
            model, shared_val_dataset, val_type, parameters
        )
        y_acc = dataloading_utils.get_acc(y_preds, y_labels)
        z_acc = dataloading_utils.get_acc(z_preds, z_labels)
        print(f"Val Type {val_type} y_acc: {100*y_acc:.2f}% z_acc: {100*z_acc:.2f}%")
    return y_acc, z_acc


def train_model(
    model: MapperModel,
    y_dataloader,
    z_dataloader,
    shared_val_dataset,
    save_path,
    load_weights,
    parameters: MapperTrainingParameters,
):
    n_epochs = parameters.total_epochs
    if load_weights and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(
            f"Loaded weights from {save_path}. Will continue for {n_epochs} more epochs"
        )
    optimizer = torch.optim.NAdam(
        [
            {
                "params": itertools.chain(
                    model.auxilary_params,
                ),
                "lr": parameters.lr,
                "weight_decay": 1e-4,
            },
            {
                "params": itertools.chain(
                    model.pretrained_params,
                ),
                "lr": parameters.lr_fine_tune,
                "weight_decay": 1e-4,
            }
        ]
    )

    def interpolate_geometric(low:float, high:float, dist:float):
        """
        Find a the point dist/100 percent of the way from low to high
        on a log scale.
        
        Dist is a parameter between
        0 and 1 indicating how close to low/high it should be.
        """
        loghi = np.log(high)
        loglow = np.log(low)
        logmid = loglow * (1 - dist) + loghi * dist
        return np.exp(logmid)

    def linear(epoch):
        return epoch / parameters.total_epochs

    def linear_2_phase(end_ratio, epoch):
        phase_1_epochs = parameters.lr_warmup_epochs
        phase_2_epochs = parameters.total_epochs - parameters.lr_warmup_epochs
        base_value = linear(epoch)
        if epoch <= phase_1_epochs:
            return base_value
        else:
            return base_value * interpolate_geometric(1, end_ratio, (epoch - phase_1_epochs) / max(1, phase_2_epochs))
        
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            lambda epoch:linear(epoch),
            partial(linear_2_phase, parameters.lr_fine_tune / parameters.lr)
        ]
    )
    best_validation_acc = 0
    valid_acc = 0
    pbar = range(0, n_epochs)
    if parameters.tqdm:
        pbar = tqdm(range(0, n_epochs),
            desc="Training epochs",
        )
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), save_path)
    for epoch_index in pbar:
        print(f"Epoch {epoch_index + 1}/{n_epochs}")
        train_epoch(
            y_dataloader, z_dataloader, model, optimizer, parameters, epoch_index
        )
        if shared_val_dataset is not None:
            valid_acc_y, valid_acc_z = model_validation_acc(
                model, shared_val_dataset, epoch_index, parameters, do_others=False
            )
            valid_acc = math.sqrt(valid_acc_y * valid_acc_z)  # Geometric Mean
            # if valid_acc > best_validation_acc:
                # Find the other statistics for this interesting model
            model_validation_acc(
                model, shared_val_dataset, epoch_index, parameters, do_others=True
            )
        if valid_acc < 0.2:
            print(f"Model collapsed, restarting from best epoch.")
            model.load_state_dict(torch.load(save_path))
        elif valid_acc >= best_validation_acc:
            best_validation_acc = valid_acc
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            print("Saving this model")
            torch.save(model.state_dict(), save_path)
        scheduler.step()
    return model


def main(
    y_dataset_name,
    z_dataset_name,
    model_name,
    load_cached_weights=True,
    parameters=None,
):
    if parameters == None:
        parameters = MapperTrainingParameters()
    save_path = Path("models") / (
        model_name.split("/")[-1] + "_mapper_" + y_dataset_name + "_" + z_dataset_name
    )
    y_dataset = training.get_dataset(y_dataset_name, "unshared")
    z_dataset = training.get_dataset(z_dataset_name, "unshared")
    y_dataloader = training.get_dataloader(
        model_name, y_dataset, parameters.batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model_name, z_dataset, parameters.batch_size, shuffle=True
    )
    model = MapperModel(
        model_name, y_dataset.num_labels, z_dataset.num_labels, parameters
    )
    model.to(parameters.device)
    shared_val_dataset = None
    if y_dataset_name == "tweebank" and z_dataset_name == "ark":
        shared_val_dataset = create_tweebank_ark_dataset()
    mapped_model = train_model(
        model,
        y_dataloader,
        z_dataloader,
        shared_val_dataset,
        save_path,
        load_cached_weights,
        parameters,
    )
    return mapped_model


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large", load_cached_weights=False,
         parameters=MapperTrainingParameters(tqdm=True))
