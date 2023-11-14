from pathlib import Path
import click
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from src.models.mobilevit import MobileVIT

CONFIG_FILE = "config/fluorescent_mobilevit_hps.yaml"
CHECKPOINT_DIR = "models/FluorescentMobileVIT/epoch=59-step=6120.ckpt"
# Checkpoint of the model used in the projec
MODEL_CHECKPOINT = "apple/deeplabv3-mobilevit-xx-small"
IMAGES_PATH = "data/processed"

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
@click.argument(
    "image_dir", type=click.Path(exists=True, file_okay=True, path_type=Path)
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default=IMAGES_PATH,
)
def single_prediction(image_dir, output_path):
    # Instantiate the model from the checkpoint and using the hparams file
    mobilevit_model = MobileVIT.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_DIR,
        hparams_file=CONFIG_FILE,
    )
    # Instantiate the image_processor
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_CHECKPOINT, do_reduce_labels=False
    )
    # Load the image
    image = Image.open(image_dir).convert("RGB")
    click.echo(f"\tLOADED: {image_dir}")
    # Convert the image to numpy array
    np_image = np.asarray(image, dtype=np.uint8)
    # Preprocess the image and move the image to the GPU Device
    processed_image = image_processor(images=np_image, return_tensors="pt")
    processed_image.to(DEVICE)
    # Make the prediction and resize the predicted mask
    click.echo(f"\tSTART PREDICTION...")
    logits = mobilevit_model.model(pixel_values=processed_image["pixel_values"])
    post_processed_image = image_processor.post_process_semantic_segmentation(
        outputs=logits, target_sizes=[(np_image.shape[0], np_image.shape[1])]
    )
    click.echo(f"\tPREDICTION FINISHED")
    try:
        image.save(f"{output_path}/image.png")
        click.echo(f"\tSAVED: {output_path}/image.png")

        mask = post_processed_image[0].data.cpu().numpy().astype(np.uint8) * 255
        click.echo(mask.max())
        mask = Image.fromarray(mask)
        mask.save(f"{output_path}/mask.png")
        click.echo(f"\tSAVED: {output_path}/mask.png")
    except Exception as e:
        click.echo(f"\tERROR: {e}")
