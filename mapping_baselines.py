import training
import numpy as np
import os
import dataloading_utils
from EncoderDecoderDataloaders import create_tweebank_ark_dataset


def normal_model_baseline(train_dataset_name, model_name):
    (twee_shared, ark_shared) = create_tweebank_ark_dataset()
    if train_dataset_name == 'ark':
        val_dataset = ark_shared
    elif train_dataset_name == 'tweebank':
        val_dataset = twee_shared
    else:
        raise NotImplementedError
    hparams = {
        "n_epochs": 10,
        "batch_size": 32,
        "dataset": train_dataset_name,
        "model_name": model_name,
    }
    hparams["save_path"] = os.path.join(
        "models",
        hparams["model_name"].split("/")[-1] + "_" + hparams["dataset"],
    )
    trained_model = training.pipeline(hparams, load_weights=True,
        use_unshared=True)
    val_dataloader = training.get_dataloader(
        model_name, val_dataset, hparams["batch_size"], shuffle=False
    )

    # validation
    preds, labels = training.validation_epoch(trained_model, val_dataloader)
    return dataloading_utils.get_acc(preds, labels)

# 3 Epochs:
# On ark: 93.5943%
# Accuracy on tweebank: 94.1281%
# 10 epochs:
# 
# Accuracy on ark: 94.3950%                        
# Accuracy on tweebank: 94.7509% (94.7954% on second run)                           
def main_normal_model():
  ark_acc = normal_model_baseline("tweebank", "vinai/bertweet-large")
  print(f"Accuracy on ark: {100*ark_acc:.4f}%")
  #twee_acc = normal_model_baseline("ark", "vinai/bertweet-large")
  #print(f"Accuracy on tweebank: {100*twee_acc:.4f}%")

if __name__ == '__main__':
  main_normal_model()
