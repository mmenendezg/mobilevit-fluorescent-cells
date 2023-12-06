import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from models.mobilevit import MobileVIT

# Checkpoint of the model used in the projec
MODEL_CHECKPOINT = "mmenendezg/mobilevit-fluorescent-neuronal-cells"

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


def single_prediction(image):
    # Instantiate the model from the checkpoint and using the hparams file
    mobilevit_model = MobileVIT()
    mobilevit_model.to(DEVICE)
    # Instantiate the image_processor
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_CHECKPOINT, do_reduce_labels=False
    )
    # Load the image
    image = image.convert("RGB")
    # Convert the image to numpy array
    np_image = np.asarray(image, dtype=np.uint8)
    # Preprocess the image and move the image to the GPU Device
    processed_image = image_processor(images=np_image, return_tensors="pt")
    processed_image.to(DEVICE)
    # Make the prediction and resize the predicted mask
    logits = mobilevit_model.model(pixel_values=processed_image["pixel_values"])
    post_processed_image = image_processor.post_process_semantic_segmentation(
        outputs=logits, target_sizes=[(np_image.shape[0], np_image.shape[1])]
    )
    # Process the mask
    mask = post_processed_image[0].data.cpu().numpy().astype(np.uint8) * 255
    mask = Image.fromarray(mask)

    return mask
