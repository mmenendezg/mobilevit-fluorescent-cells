import os
import yaml
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .mobilevit import MobileVIT
from ..data.data_preprocessing import FluorescentNeuronalDataModule


MODEL_CHECKPOINT = "apple/deeplabv3-mobilevit-xx-small"

# Define the accelerator
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps:0")
    ACCELERATOR = "mps"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    ACCELERATOR = "gpu"
else:
    DEVICE = torch.device("cpu")
    ACCELERATOR = "cpu"

RAW_DATA_PATH = "../data/raw/"
DATA_PATH = "../data/processed/"

CLASSES = {0: "Background", 1: "Neuron"}

IMG_SIZE = [256, 256]


def objective(trial: optuna.Trial, dataset_size: float = 0.25) -> float:
    """
    DOCSTRING
    """
    # Suggest values of the hyperparameters for the trials
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 2, 4, log=True)

    # Define the callbacks of the model
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=2)

    # Create the model
    model = MobileVIT(learning_rate=learning_rate, weight_decay=weight_decay)

    # Instantiate the data module
    data_module = FluorescentNeuronalDataModule(
        batch_size=batch_size, dataset_size=dataset_size
    )
    data_module.setup()

    # Train the model
    trainer = pl.Trainer(
        devices=1,
        accelerator=ACCELERATOR,
        precision="16-mixed",
        max_epochs=5,
        log_every_n_steps=5,
        callbacks=[early_stopping_cb],
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    return trainer.callback_metrics["val_loss"].item()


def get_best_params(config_file: str, force_tune: bool = False) -> dict:
    if os.path.exists(config_file) and force_tune:
        os.remove(config_file)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)

        study.optimize(objective, n_trials=25)
        best_params = study.best_params
        with open(config_file, "w") as file:
            yaml.dump(best_params, file)
    elif os.path.exists(config_file):
        with open(config_file, "r") as file:
            best_params = yaml.safe_load(file)
    else:
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)

        study.optimize(objective, n_trials=25)
        best_params = study.best_params
        with open(config_file, "w") as file:
            yaml.dump(best_params, file)

    return best_params
