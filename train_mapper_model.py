from augmented_datasets import get_dataloader
from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset, flatten_preds_and_labels
import torch
from tqdm import tqdm
import training
from typing import Tuple
import torch.nn.functional as F
from pathlib import Path
import math
import os
from typing import List
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
from transformers import get_scheduler

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
    y_pred = F.softmax(y_tilde, dim=2)
    loss = torch.nn.CrossEntropyLoss()  # This line might be wrong
    return loss(y_pred.flatten(0, 1), labels.flatten())


def train_epoch(
    y_dataloader: TransformerCompatDataset,
    z_dataloader: TransformerCompatDataset,
    model: MapperModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
):
    model.train()
    n_iters = min(len(y_dataloader), len(z_dataloader))
    sum_train_loss = 0
    # Iterate until either dataset is exhausted.
    for batch_y, batch_z in tqdm(zip(y_dataloader, z_dataloader),total=n_iters):
        loss = compose_loss(batch_y, model, "y") + compose_loss(batch_z, model, "z")
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        sum_train_loss = loss.detach()
    return sum_train_loss / n_iters


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
    optimizer = torch.optim.NAdam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=min(len(y_dataloader), len(z_dataloader)) * n_epochs,
    )
    best_validation_acc = 0
    valid_acc = 0
    #if shared_val_dataset is not None:
    #     Test
    #    valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs",):
        train_loss = train_epoch(
            y_dataloader, z_dataloader, model, optimizer, scheduler
        )
        print(f"Train KL Loss: {train_loss}")
        if shared_val_dataset is not None:
            valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
            valid_acc = math.sqrt(valid_acc_y * valid_acc_z) # Geometric Mean
            print(f"Val Acc Y: {valid_acc_y*100}% Val Acc Z {valid_acc_z*100}% Soft label {model.soft_label_value}")
        if valid_acc >= best_validation_acc:
            best_validation_acc = valid_acc
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(model.state_dict(), save_path)
    return model



def main(y_dataset_name, z_dataset_name, model_name):
    n_epochs = 10
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
        load_weights=True
    )
    # After 10 epochs:
    # Val Acc Y: 92.12633451957295% Val Acc Z 92.48220640569394%


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large")
