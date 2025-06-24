import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=10, img_size=28):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, nz)
        self.model = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embed = self.label_embed(labels)
        x = z * embed
        return self.model(x).view(-1, 1, 28, 28)

# Dummy training loop (you can replace this with your real training loop)
def train_and_save_model():
    model = Generator()
    # Skip actual training; save untrained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/generator.pth")

if __name__ == "__main__":
    train_and_save_model()