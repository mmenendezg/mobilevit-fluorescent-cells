import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import MobileViTForSemanticSegmentation
import evaluate

MODEL_CHECKPOINT = "mmenendezg/mobilevit-fluorescent-neuronal-cells"
CLASSES = {0: "Background", 1: "Neuron"}


class MobileVIT(pl.LightningModule):
    def __init__(self, learning_rate=None, weight_decay=None):
        super().__init__()
        self.id2label = CLASSES
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_classes = len(self.id2label.keys())
        self.model = MobileViTForSemanticSegmentation.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.metric = evaluate.load("mean_iou")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, pixel_values, labels):
        return self.model(pixel_values=pixel_values, labels=labels)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

    def compute_metric(self, logits, labels):
        logits_tensor = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = self.metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        return metrics

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]

        # Calculate and log the loss
        loss, logits = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)

        # Calculate and log the metrics
        metrics = self.compute_metric(logits, labels)
        metrics = {key: np.float32(value) for key, value in metrics.items()}

        self.log("train_mean_iou", metrics["mean_iou"])
        self.log("train_mean_accuracy", metrics["mean_accuracy"])
        self.log("train_overall_accuracy", metrics["overall_accuracy"])

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]

        # Calculate and log the loss
        loss, logits = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)

        # Calculate and log the metrics
        metrics = self.compute_metric(logits, labels)
        metrics = {key: np.float32(value) for key, value in metrics.items()}
        self.log("val_mean_iou", metrics["mean_iou"])
        self.log("val_mean_accuracy", metrics["mean_accuracy"])
        self.log("val_overall_accuracy", metrics["overall_accuracy"])

        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]

        # Calculate and log the loss
        loss, logits = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)

        # Calculate and log the metrics
        metrics = self.compute_metric(logits, labels)
        metrics = {key: np.float32(value) for key, value in metrics.items()}
        # for k, v in metrics.items():
        #     self.log(f"val_{k}", v.item())
        self.log("test_mean_iou", metrics["mean_iou"])
        self.log("test_mean_accuracy", metrics["mean_accuracy"])
        self.log("test_overall_accuracy", metrics["overall_accuracy"])

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters()],
                "lr": self.learning_rate,
            }
        ]
        return torch.optim.AdamW(
            param_dicts, lr=self.learning_rate, weight_decay=self.weight_decay
        )
