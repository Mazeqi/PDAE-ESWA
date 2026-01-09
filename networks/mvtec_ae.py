import torch.nn as nn
import torch
 
class Encoder(nn.Module):
    def __init__(self, latent_dim = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False),
        )


    def forward(self, img):
        features = self.model(img)
        return features


class Decoder(nn.Module):
    def __init__(self, img_shape = (3, 256, 256), latent_dim = 100):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.model(input)
        return img.view(img.shape[0], *self.img_shape)

class CAE(nn.Module):
    """This class implements the Convolutional AutoEncoder for normal image generation
    CAE is processed in encoder and decoder that is composed CNN layers
    """
    def __init__(self, image_shape = (3, 256, 256), latent_dim = 100):
        """Initialize the CAE model"""
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(image_shape, latent_dim)


    def forward(self, real_imgs):
        features = self.encoder(real_imgs)
        generated_imgs = self.decoder(features)

        return generated_imgs
    

