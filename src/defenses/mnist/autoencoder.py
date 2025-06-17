import torch
import torch.nn as nn

class AutoencoderDenoiserMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER: 28→14→7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # [B,8,14,14]
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), # [B,16,7,7]
            nn.ReLU(True),
        )
        # DECODER: 7→14→28, dengan output_padding=1 pada masing-masing
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), # [B,8,14,14]
            nn.ReLU(True),
            nn.ConvTranspose2d(8,  1, 3, stride=2, padding=1, output_padding=1), # [B,1,28,28]
            nn.Sigmoid()  # mengembalikan nilai 0–1 sesuai ToTensor()
        )

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon