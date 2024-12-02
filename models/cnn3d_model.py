import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNN3D(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001):
        super(CNN3D, self).__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 128)  # 4 for the one-hot encoded camera angle
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, angle):
        """
        Forward pass through the network.

        Args:
            x: A tensor of shape (batch_size, 3, sequence_length, height, width).
            angle: A tensor of shape (batch_size, 4) representing the one-hot encoded camera angle.

        Returns:
            A tensor of shape (batch_size, num_classes) representing class logits.
        """
        x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool(F.relu(x))
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool(F.relu(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # # print(x.shape)
        # x = torch.cat((x, angle), dim=1)  # Concatenate angle information
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: A tuple (sequences, labels, angles).
            batch_idx: Index of the current batch.

        Returns:
            Loss value for the current batch.
        """
        sequences, labels, angles = batch
        outputs = self(sequences, angles)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: A tuple (sequences, labels, angles).
            batch_idx: Index of the current batch.

        Returns:
            Loss value for the current batch.
        """
        sequences, labels, angles = batch
        outputs = self(sequences, angles)
        loss = F.cross_entropy(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers.

        Returns:
            Optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
