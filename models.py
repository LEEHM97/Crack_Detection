import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from adamp import AdamP
from torchmetrics.functional import accuracy, f1_score, precision, recall
# from torchmetrics.functional.classification import binary_jaccard_index


class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.model = smp.__dict__[args.model](
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == "adamp":
            optimizer = AdamP(
                self.parameters(),
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, mode="max", verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=True
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch

        outputs = self.model(image)
        mask = mask.long()
        
        # jaccard_index_value = binary_jaccard_index(
        #     torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3)
        # )        
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        acc_value = accuracy(torch.sigmoid(outputs), mask)
        f1_value = f1_score(torch.sigmoid(outputs), mask)
        precision_value = precision(torch.sigmoid(outputs), mask)
        recall_value = recall(torch.sigmoid(outputs), mask)

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/acc", acc_value, on_epoch=True, on_step=True, prog_bar=True)
        # self.log(
        #     "train/jaccard_index_value",
        #     jaccard_index_value,
        #     on_epoch=True,
        #     on_step=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.log(
            "train/f1",
            f1_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/precision",
            precision_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/recall",
            recall_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {
            "loss": loss,
            "acc": acc_value,
            # "jaccard_index": jaccard_index_value,
            "f1": f1_value,
            "precision": precision_value,
            "recall": recall_value,
        }

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch

        outputs = self.model(image)
        mask = mask.long()

        # jaccard_index_value = binary_jaccard_index(
        #     torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3)
        # )
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        acc_value = accuracy(torch.sigmoid(outputs), mask)
        f1_value = f1_score(torch.sigmoid(outputs), mask)
        precision_value = precision(torch.sigmoid(outputs), mask)
        recall_value = recall(torch.sigmoid(outputs), mask)

        self.log(
            "val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True
        )
        self.log("val/acc", acc_value, on_epoch=True, on_step=True, prog_bar=True)
        # self.log(
        #     "val/jaccard_index_value",
        #     jaccard_index_value,
        #     on_epoch=True,
        #     on_step=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.log(
            "val/f1",
            f1_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/precision",
            precision_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/recall",
            recall_value,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {
            "loss": loss,
            "acc": acc_value,
            # "jaccard_index": jaccard_index_value,
            "f1": f1_value,
            "precision": precision_value,
            "recall": recall_value,
        }

    def test_step(self, test_batch, batch_idx):
        image, mask = test_batch

        outputs = self.model(image)
        mask = mask.long()

        # jaccard_index_value = binary_jaccard_index(
        #     torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3)
        # )        
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        acc_value = accuracy(torch.sigmoid(outputs), mask)
        f1_value = f1_score(torch.sigmoid(outputs), mask)
        precision_value = precision(torch.sigmoid(outputs), mask)
        recall_value = recall(torch.sigmoid(outputs), mask)

        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("test/acc", acc_value, on_epoch=True, on_step=False, prog_bar=True)
        # self.log(
        #     "test/jaccard_index_value",
        #     jaccard_index_value,
        #     on_epoch=True,
        #     on_step=False,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.log(
            "test/f1",
            f1_value,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/precision",
            precision_value,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/recall",
            recall_value,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        return {
            "loss": loss,
            "acc": acc_value,
            # "jaccard_index": jaccard_index_value,
            "f1": f1_value,
            "precision": precision_value,
            "recall": recall_value,
        }
