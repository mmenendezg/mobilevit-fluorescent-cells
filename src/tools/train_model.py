import yaml
from pathlib import Path
import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.mobilevit import MobileVIT
from src.data.data_preprocessing import FluorescentNeuronalDataModule

CONFIG_FILE = "config/fluorescent_mobilevit_hps.yaml"
DATA_DIR = "data/raw/"
LOGS_DIR = "reports/logs/FluorescentMobileVIT"
MODEL_DIR = "models/FluorescentMobileVIT"

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


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default=DATA_DIR,
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default=CONFIG_FILE,
)
def train_model(data_dir, config_file):
    # Load the best parameters
    with open(config_file, "r") as file:
        best_params = yaml.safe_load(file)
    # Instantiate the model
    model = MobileVIT(
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )
    # Define the callbacks of the model
    model_checkpoint_cb = ModelCheckpoint(
        save_top_k=1, dirpath=MODEL_DIR, monitor="val_loss"
    )
    logger = TensorBoardLogger(save_dir=LOGS_DIR)

    # Create the trainer with its parameters
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        accelerator=ACCELERATOR,
        precision=16,
        max_epochs=100,
        log_every_n_steps=20,
        callbacks=[model_checkpoint_cb],
    )
    data_module = FluorescentNeuronalDataModule(
        data_dir=data_dir, batch_size=best_params["batch_size"]
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    click.echo("\n\n==========The Training has Finished!==========")
