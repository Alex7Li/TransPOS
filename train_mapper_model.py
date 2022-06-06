from augmented_datasets import get_dataloader
from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset
import torch
from tqdm import tqdm
import training
from typing import Tuple
from pathlib import Path
import numpy as np
from EncoderDecoderDataloaders import create_tweebank_ark_dataset
from transformers import get_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: UNTESTED
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
    loss = torch.nn.KLDivLoss()  # This line might be wrong
    return loss(y_tilde, batch["labels"])


# TODO: UNTESTED
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
    for i in tqdm(range(n_iters), desc="Epoch progress", mininterval=5):
        for batch_y, batch_z in zip(y_dataloader, z_dataloader):
            loss = compose_loss(batch_y, model, "y")
            loss += compose_loss(batch_z, model, "z")
            loss.backward()
            sum_train_loss = loss.detach()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return sum_train_loss / n_iters


def get_validation_predictions(model: MapperModel, shared_val_set):
    model.eval()
    y_dataset, z_dataset = shared_val_set
    batch_size = 32
    y_dataloader = training.get_dataloader(
        model.base_transformer_name, y_dataset, batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model.base_transformer_name, z_dataset, batch_size, shuffle=True
    )
    predicted_y = []
    predicted_z = []
    for y_batch, z_batch in tqdm(
        zip(y_dataloader, z_dataloader),
        desc="Predicting Validation labels",
        mininterval=5,
        total=len(y_dataloader),
    ):
        e = model.encode(y_batch)
        z_pred = torch.argmax(model.decode_y(e, y_batch['labels'].to(device)), dim=2).flatten()
        y_pred = torch.argmax(model.decode_z(e, z_batch['labels'].to(device)), dim=2).flatten()
        predicted_z.append(z_pred)
        predicted_y.append(y_pred)
    return np.stack(predicted_y, axis=0), np.stack(predicted_z, axis=0)

def get_validation_shared(y_predictions: np.ndarray, z_predictions: np.ndarray, shared_val_datasets) -> Tuple[float, float]:
    """
    Get the validation accuracy from the predictions.
    y_predictions[i]: The predicted y labels for the ith element of the valdation dataloader
    z_predictions[i]: The predicted z labels for the jth element of the valdation dataloader
    shared_val_dataloaders: A tuple (y_dataloader, z_dataloader)
    """
    y_correct = 0
    z_correct = 0
    n_examples = len(shared_val_datasets[0])
    total = 0
    for i in range(n_examples):
        y_correct += np.sum(y_predictions[i] == shared_val_datasets[0][i] & shared_val_datasets[0][i] != -100)
        z_correct += np.sum(z_predictions[i] == shared_val_datasets[1][i] & shared_val_datasets[1][i] != -100)
        total += np.sum(shared_val_datasets[1][i] != -100)
    return y_correct / total, z_correct / total


def model_validation_acc(model: MapperModel, shared_val_dataset) -> Tuple[float, float]:
    predicted_y, predicted_z = get_validation_predictions(model, shared_val_dataset)
    return get_validation_shared(predicted_y, predicted_z, shared_val_dataset)


def train_model(
    model: MapperModel,
    y_dataloader,
    z_dataloader,
    shared_val_dataset,
    n_epochs,
    save_path,
):
    optimizer = torch.optim.NAdam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=min(len(y_dataloader), len(z_dataloader)) * n_epochs,
    )
    best_validation_acc = 0
    valid_acc = 0
    if shared_val_dataset is not None:
        # Test
        valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs", maxinterval=50):
        train_loss = train_epoch(
            y_dataloader, z_dataloader, model, optimizer, scheduler
        )
        print(f"Train KL Loss: {train_loss}")
        if shared_val_dataset is not None:
            valid_acc_y, valid_acc_z = model_validation_acc(model, shared_val_dataset)
            valid_acc = np.sqrt(valid_acc_y * valid_acc_z) # Geometric Mean
            print(f"Val Acc Y: {valid_acc_y} Val Acc Z {valid_acc_z} ", end=None)
        if valid_acc >= best_validation_acc:
            best_validation_acc = valid_acc
            torch.save(model.state_dict(), save_path)
    return model


def main(y_dataset_name, z_dataset_name, model_name):
    batch_size = 32
    n_epochs = 5
    save_path = Path("models") / (
        model_name.split("/")[-1] + "_mapper_" + y_dataset_name + "_" + z_dataset_name
    )
    y_dataset = training.get_dataset(y_dataset_name, "all")
    z_dataset = training.get_dataset(z_dataset_name, "all")
    y_dataloader = training.get_dataloader(
        model_name, y_dataset, batch_size, shuffle=True
    )
    z_dataloader = training.get_dataloader(
        model_name, z_dataset, batch_size, shuffle=True
    )
    model = MapperModel(
        "vinai/bertweet-large", y_dataset.num_labels, z_dataset.num_labels
    )
    shared_val_dataset = None
    if y_dataset_name == "tweebank" and z_dataset_name == "ark":
        shared_val_dataset = create_tweebank_ark_dataset()
    train_model(
        model, y_dataloader, z_dataloader, shared_val_dataset, n_epochs, save_path
    )


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large")
