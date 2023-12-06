import os
import gradio as gr
from tools.predict import single_prediction

KAGGLE_NOTEBOOK = "[![Static Badge](https://img.shields.io/badge/Open_Notebook_in_Kaggle-gray?logo=kaggle&logoColor=white&labelColor=20BEFF)](https://www.kaggle.com/code/mmenendezg/mobilevit-fluorescent-neuronal-cells/notebook)"
GITHUB_REPOSITORY = "[![Static Badge](https://img.shields.io/badge/Git_Repository-gray?logo=github&logoColor=white&labelColor=181717)](https://github.com/mmenendezg/mobilevit-fluorescent-cells)"
HF_SPACE = "[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/mmenendezg/mobilevit-fluorescent-neuronal-cells)"

# Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown(
        f"""
    # Fluorescent Neuronal Cells Segmentation

    This model extracts a segmentation mask of the neuronal cells on an image.

    {KAGGLE_NOTEBOOK}

    {GITHUB_REPOSITORY}

    {HF_SPACE}
    """
    )
    with gr.Tab("Image Segmentation"):
        with gr.Row():
            with gr.Column():
                uploaded_image = gr.Image(
                    label="Neuronal Cells Image",
                    sources=["upload", "clipboard"],
                    type="pil",
                    height=550,
                )
            with gr.Column():
                mask_image = gr.Image(label="Segmented Neurons", height=550)
        with gr.Row():
            classify_btn = gr.Button("Segment the image", variant="primary")
            clear_btn = gr.ClearButton(components=[uploaded_image, mask_image])
        classify_btn.click(
            fn=single_prediction, inputs=uploaded_image, outputs=[mask_image]
        )
        gr.Examples(
            examples=[
                os.path.join(os.path.dirname(__file__), "examples/example_1.png"),
                os.path.join(os.path.dirname(__file__), "examples/example_2.png"),
            ],
            inputs=uploaded_image,
        )
demo.launch(show_error=True)
