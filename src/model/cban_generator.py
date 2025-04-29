import torch
import torch.nn as nn

class CBaNGenerator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, output_dim=28*28):
        super(CBaNGenerator, self).__init__()
        # dua input layer: noise(z) dan label(l)
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU()
        )
        self.label_fc = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.ReLU()
        )
        
        # digabung lalu dilewatkan ke jaringan utama
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128), # input = concat z + l
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # karena output trigger perlu dalam range [0,1]
        )

    def forward(self, noise, labels):
        z = self.noise_fc(noise)
        l = self.label_fc(labels)
        x = torch.cat([z, l], dim=1) # gabung di dimensi fitur (bukan batch)
        x = self.fc(x)
        return x.view(-1, 1, 28, 28)  # reshape jadi [batch_size, 1, 28, 28] (ukuran MNIST)
        # return x.view(-1, 1, int((x.shape[1])**0.5), int((x.shape[1])**0.5)) # dynamically reshape (after done mnist development)


        """
        z:      [B, 100] ─> FC ─> [B, 64]
        label:  [B, 10 ] ─> FC ─> [B, 64]
                            ↓
                    cat([z,l]) ─> [B, 128]
                            ↓
                        FC ─> FC ─> FC ─> FC
                            ↓
                        output: [B, 1, 28, 28]            
        """