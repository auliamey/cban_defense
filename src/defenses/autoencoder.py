import torch
import torch.nn as nn

class AutoencoderDenoiser(nn.Module):
    def __init__(self):
        super(AutoencoderDenoiser, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [B, 64, 4, 4]
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # [B, 32, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [B, 16, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # [B, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
