from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
def train_model():
    """
    Train the MobileViT model on a new dataset.
    """
    pass


@cli.command()
def get_best_params():
    """
    Get the best hyperparameters to train the model.
    """
    pass


@cli.command()
@click.argument(
    "image-path", type=click.STRING
)
def single_prediction():
    """
    Make a prediction for a single image.

    IMAGE_PATH is the path to the image for the prediction.
    """
    pass


@cli.command()
@click.argument(
    "file_path",
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
def batch_prediction():
    """
    Make a prediction for the images in a folder.

    FILE_PATH is the path to the folder that contains the images for the prediction.
    """
    pass


if __name__ == "__main__":
    cli()
