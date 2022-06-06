from torchmetrics import KLDivergence
from augmented_datasets import get_dataloader
from mapper_model import MapperModel
import dataloading_utils
from dataloading_utils import TransformerCompatDataset
import torch
from tqdm import tqdm
import training
from pathlib import Path
from EncoderDecoderDataloaders import TweebankArkDataset, create_tweebank_ark_dataset
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
    e_y = model.encode(batch["output"])
    y_tilde = decode_y(e_y, batch["labels"])
    y_tilde = decode_z(e_y, y_tilde)
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
        loss = compose_loss(y_dataloader[i], model, "y")
        loss += compose_loss(z_dataloader[i], model, "z")
        loss.backward()
        sum_train_loss = loss.detach()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return sum_train_loss / n_iters


def validation_acc(model: MapperModel, shared_val_dataset) -> float:
    model.eval()
    y_dataset, z_dataset = shared_val_dataset
    for (x, y), (x, z) in tqdm(
        zip(y_dataset, z_dataset),
        desc="Validation",
        mininterval=5,
        total=len(y_dataset),
    ):
        e_y = model.encode(x)

        predictions = model()
    return 0


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
    for epoch_index in tqdm(range(0, n_epochs), desc="Training epochs", maxinterval=50):
        train_loss = train_epoch(
            y_dataloader, z_dataloader, model, optimizer, scheduler
        )
        if shared_val_dataset is not None:
            valid_acc = validation_acc(model, shared_val_dataset)
        print(f"Train loss {train_loss}, Validation acc {valid_acc}")
        if valid_acc >= best_validation_acc:
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
    if y_dataset == "tweebank" and z_dataset == "ark":
        shared_val_dataset = create_tweebank_ark_dataset
    train_model(
        model, y_dataloader, z_dataloader, shared_val_dataset, n_epochs, save_path
    )


if __name__ == "__main__":
    main("tweebank", "ark", "vinai/bertweet-large")
