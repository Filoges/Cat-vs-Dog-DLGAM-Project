import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator32(nn.Module):
    def __init__(self, nc):
        super(Generator32, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),                  #(W - 1)S -2P + (K - 1) + 1
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)



class Discriminator32(nn.Module):
    def __init__(self, nc):
        super(Discriminator32, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class Generator64(nn.Module):
    def __init__(self, nc):
        super(Generator64, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),                  #(W - 1)S -2P + (K - 1) + 1
            nn.BatchNorm2d(ngf * 8, momentum=0.7),
            # nn.ReLu(True),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.7),
            # nn.ReLu(True),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.7),
            # nn.ReLu(True),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.7),
            # nn.ReLu(True),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    


class Discriminator64(nn.Module):
    def __init__(self, nc):
        super(Discriminator64, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class Generator128(nn.Module):
    def __init__(self, nc):
        super(Generator128, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16, momentum=0.7),
            nn.ReLU(True),
            #jjjj
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator128(nn.Module):
    def __init__(self, nc):
        super(Discriminator128, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


class Generator256(nn.Module):
    def __init__(self, nc):
        super(Generator256, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),     # Output size: (ngf*16) x 4 x 4
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size: (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),  # Output size: (ngf*8) x 8 x 8
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size: (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),   # Output size: (ngf*4) x 16 x 16
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),   # Output size: (ngf*2) x 32 x 32
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),       # Output size: (ngf) x 64 x 64
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf  * 2, ngf, 4, 2, 1, bias=False),           # Output size: (nc) x 128 x 128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),            # Output size: (nc) x 256 x 256
            nn.Tanh()
            # state size: (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator256(nn.Module):
    def __init__(self, nc):
        super(Discriminator256, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        hidden_size=64,
        latent_size=nz
    ):
        super(VAE, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        ## encoder ##
        self.Encoder = nn.Sequential(nn.Conv2d(image_channels, 16, 4, stride = 2, padding=1), #8x32x32   [(W−K+2P)/S]+1
                                        nn.ReLU(),
                                        nn.Conv2d(16, 32, 4, stride = 2, padding=1), #32x8x8
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride = 2, padding=1), #64x4x4
                                        nn.ReLU(),
                                        nn.Conv2d(64, self.hidden_size, 4, stride = 1, padding=0), #64x1x1
                                        nn.ReLU(), 
                                        nn.Flatten())  #hidden_size
        

        # define mean
        self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
        #define logvar
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        self.resize_fc = nn.Linear(self.latent_size, self.hidden_size)
        
        ## decoder ##
        self.Decoder = nn.Sequential(nn.ConvTranspose2d( self.hidden_size, 64, 4, 1, 0), #64x4x4  (W - 1)S -2P + (K - 1) + 1
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 64, 32, 4, 2, 1), #32x8x8
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 32, 16, 4, 2, 1), #16x16x16
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 16, image_channels, 4, 2, 1), #8x32x32
                                    nn.Sigmoid())

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean) # reparametrization trick
    
    def KL_loss(self, log_var, mean):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        x = self.Encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        
        z = self.sample(log_var, mean)
        
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)
        
        x = self.Decoder(x)

        return x, log_var, mean
    
    def generate_img(self, z):
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)    
        return self.Decoder(x)


class VAE64(nn.Module):
    def __init__(
        self,
        image_channels=3,
        hidden_size=64,
        latent_size=nz
    ):
        super(VAE64, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        ## encoder ##
        self.Encoder = nn.Sequential(nn.Conv2d(image_channels, 8, 4, stride = 2, padding=1), #8x32x32   [(W−K+2P)/S]+1
                                        nn.ReLU(),
                                        nn.Conv2d(8, 16, 4, stride = 2, padding=1), #8x32x32   [(W−K+2P)/S]+1
                                        nn.ReLU(),
                                        nn.Conv2d(16, 32, 4, stride = 2, padding=1), #32x8x8
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride = 2, padding=1), #64x4x4
                                        nn.ReLU(),
                                        nn.Conv2d(64, self.hidden_size, 4, stride = 1, padding=0), #64x1x1
                                        nn.ReLU(), 
                                        nn.Flatten())  #hidden_size
        

        # define mean
        self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
        #define logvar
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        self.resize_fc = nn.Linear(self.latent_size, self.hidden_size)
        
        ## decoder ##
        self.Decoder = nn.Sequential(nn.ConvTranspose2d( self.hidden_size, 64, 4, 1, 0), #64x4x4  (W - 1)S -2P + (K - 1) + 1
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 64, 32, 4, 2, 1), #32x8x8
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 32, 16, 4, 2, 1), #16x16x16
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 16, 8, 4, 2, 1), #8x32x32
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 8, image_channels, 4, 2, 1), #8x32x32
                                    nn.Sigmoid())

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean) # reparametrization trick
    
    def KL_loss(self, log_var, mean):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        x = self.Encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        
        z = self.sample(log_var, mean)
        
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)
        
        x = self.Decoder(x)

        return x, log_var, mean
    
    def generate_img(self, z):
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)    
        return self.Decoder(x)
    

class WGANGen32(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANGen32, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class WGANDis32(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANDis32, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 32 x 32
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid() NO SIGMOID IN WGAN
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    


class WGANGen64(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANGen64, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 6, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6, momentum=0.7),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class WGANDis64(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANDis64, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 32 x 32
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid() NO SIGMOID IN WGAN
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    


class WGANGen128(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANGen128, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 14, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 14, momentum=0.7, eps=0.0001),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=0.9, eps=0.0001),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.9, eps=0.0001),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.9, eps=0.0001),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.9, eps=0.0001),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class WGANDis128(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANDis128, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid() NO SIGMOID IN WGAN
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    

class WGANGen256(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANGen256, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 30, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 30),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class WGANDis256(nn.Module):
    def __init__(self, nc, label_dim):
        super(WGANDis256, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid() NO SIGMOID IN WGAN
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    
    
class CGANGen32(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANGen32, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class CGANDis32(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANDis32, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 32 x 32
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    

class CGANGen64(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANGen64, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 6, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6,  momentum=0.7),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2,  momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf,  momentum=0.7),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class CGANDis64(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANDis64, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 32 x 32
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    


class CGANGen128(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANGen128, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 14, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 14),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class CGANDis128(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANDis128, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    

class CGANGen256(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANGen256, self).__init__()
        
        self.noise_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 30, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 30),
            nn.ReLU(True)
         )
        
        self.label_block = nn.Sequential(
            # input are labels, going into a convolution
            nn.ConvTranspose2d(label_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
         )
        
        self.main = nn.Sequential(
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over channels
        return self.main(x)


#Discriminator Code 

class CGANDis256(nn.Module):
    def __init__(self, nc, label_dim):
        super(CGANDis256, self).__init__()
        
        self.img_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_block = nn.Sequential(        
            # input is (nc) x 128 x 128
            nn.Conv2d(label_dim, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )        
        self.main = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        # same steps as in generator but with images and labels
        img_out = self.img_block(img)
        lab_out = self.label_block(label)
        x = torch.cat([img_out, lab_out], dim = 1)
        return self.main(x)
    

class VAE64WithBN(nn.Module):
    def __init__(
        self,
        image_channels=3,
        hidden_size=64,
        latent_size=nz
    ):
        super(VAE64, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        ## encoder ##
        self.Encoder = nn.Sequential(nn.Conv2d(image_channels, 8, 4, stride = 2, padding=1), #8x32x32   [(W−K+2P)/S]+1
                                        nn.ReLU(),
                                        nn.Conv2d(8, 16, 4, stride = 2, padding=1), #8x32x32   [(W−K+2P)/S]+1
                                        nn.ReLU(),
                                        nn.Conv2d(16, 32, 4, stride = 2, padding=1), #32x8x8
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride = 2, padding=1), #64x4x4
                                        nn.ReLU(),
                                        nn.Conv2d(64, self.hidden_size, 4, stride = 1, padding=0), #64x1x1
                                        nn.ReLU(), 
                                        nn.Flatten())  #hidden_size
        

        # define mean
        self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
        #define logvar
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        self.resize_fc = nn.Linear(self.latent_size, self.hidden_size)
        
        ## decoder ##
        self.Decoder = nn.Sequential(nn.ConvTranspose2d( self.hidden_size * 8, ngf * 4, 4, 1, 0), #64x4x4  (W - 1)S -2P + (K - 1) + 1
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 64, 32, 4, 2, 1), #32x8x8
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 32, 16, 4, 2, 1), #16x16x16
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 16, 8, 4, 2, 1), #8x32x32
                                    nn.ReLU(),
                                    nn.ConvTranspose2d( 8, image_channels, 4, 2, 1), #8x32x32
                                    nn.Sigmoid())

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean) # reparametrization trick
    
    def KL_loss(self, log_var, mean):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        x = self.Encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        
        z = self.sample(log_var, mean)
        
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)
        
        x = self.Decoder(x)

        return x, log_var, mean
    
    def generate_img(self, z):
        x = self.resize_fc(z).view(z.size(0),self.hidden_size,1,1)    
        return self.Decoder(x)