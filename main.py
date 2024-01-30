

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.utils as vutils
from torchvision import transforms
import matplotlib.pyplot as plt
import os

import dataloaders
import models

import json

# from torch.utils.tensorboard import SummaryWriter

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 500

# size of labels
label_dim = 2

# Learning rate for optimizers
gen_learning_rate = 0.0002
dis_learning_rate = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

momentum = 0.9

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Size of random vector --> How many fake images to generate in order to check generator results
test_size = 16

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

TRAIN_FOLDER1 = "./dataset-iiit-pet-master/images/"
TRAIN_FOLDER2 = "./PetImages/Dog/"
TRAIN_FOLDER3 = "./PetImages/Cat/"
TRAIN_FOLDER4 = "./Cat dataset with annotation/dataset/data/"
TRAIN_FOLDER5 = "./Dog dataset with annotation/Images/"
TRAIN_FOLDER6 = "./Cat dataset/cat-dataset-master/data/clean/"
TRAIN_FOLDER7 = "./Dog faces/dataset/data/"

LABEL_FOLDER = "./dataset-iiit-pet-master/annotations/list.txt"
PATH_TO_SAVE = "Models/Default/"

# Oggetto per scrivere i dati su TensorBoard
# writer = SummaryWriter(log_dir=os.path.join(PATH_TO_SAVE, "Tensorboard"))

imagenet_mean = (0.5, 0.5, 0.5)
imagenet_std = (0.5, 0.5, 0.5)


def set_seed():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    # print("Random Seed: ", manualSeed)
    # random.seed(manualSeed)
    torch.manual_seed(manualSeed)


# Metodo per la creazione del dataset per il variational autoencoder. 
def data_loading_VA(dataset_type, dataset):

    if dataset == 'ANIMALS':

        data_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])
        
        augmented_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=90),  
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])

        #train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
        #train_data2 = dataloaders.CustomDataset(root=TRAIN_FOLDER2, transform=train_tfm, label=1)
        #train_data3 = dataloaders.CustomDataset(root=TRAIN_FOLDER3, transform=train_tfm, label=0)
        #train_data2 = torchvision.datasets.ImageFolder(root=TRAIN_FOLDER2, transform=train_tfm)
        #train_data4 = dataloaders.CustomDataset(root=TRAIN_FOLDER1, transform=augm_train, path_to_id=LABEL_FOLDER)
        if dataset_type == 1:
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=data_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)

            train_data = torch.utils.data.ConcatDataset([train_data5, train_data6, train_data7])

        elif dataset_type == 2:
            train_data = dataloaders.CatFaces(root_dir=TRAIN_FOLDER4, transform=data_transform)

        train_dataset, test_dataset = torch.utils.data.random_split(train_data, [0.8, 0.2])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True)
    
    elif dataset == 'MNIST':
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=(0.5,), std=(0.5,))
                                        ])

        mnist_data_train = torchvision.datasets.MNIST(root='./MNIST',
                                train=True,
                                transform=transform,
                                download=False)
        
        mnist_data_test = torchvision.datasets.MNIST(root='./MNIST',
                                train=False,
                                transform=transform,
                                download=False)

        train_loader = torch.utils.data.DataLoader(dataset=mnist_data_train,
                                                batch_size=batch_size,
                                                shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(dataset=mnist_data_test,
                                                batch_size=batch_size,
                                                shuffle=True)

    return train_loader, test_loader

# Metodo per la creazione del dataset per le GAN.
def data_loading(dataset_type, dataset):

    if dataset == 'ANIMALS':

        data_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))])
        
        augmented_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=90),  
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))])

        # Trasformazioni che usano Albumentations. Facendo diversi testi mi sono trovato meglio con le trasformazioni
        # di pyTorch.
        train_tfm = A.Compose([
            A.Resize(image_size, image_size,  interpolation=1, always_apply=True, p=1),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean = imagenet_mean, std = imagenet_std),
            ToTensorV2(),
        ])

        augm_train = A.Compose([
            A.Resize(image_size, image_size,  interpolation=1, always_apply=True, p=1),
            A.CenterCrop(image_size, image_size),
            #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=0.5), 
            A.RandomRotate90(),
            A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.5),  
            A.Normalize(mean = imagenet_mean, std = imagenet_std),
            ToTensorV2(),
        ])
        

        # Qui si compone il dataset in base al valore di "dataset_type" nel file json. 
        # Il primo caso usa solo il dataset originale.
        if dataset_type == 1:
            train_data = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)

            # to_pil = transforms.ToPILImage()
            # labels_map = {
            #     0: "Cat",
            #     1: "Dog"
            # }
            # figure = plt.figure(figsize=(16, 16))
            # cols, rows = 4, 4
            # for i in range(1, cols * rows + 1):
            #     sample_idx = torch.randint(len(train_data), size=(1,)).item()
            #     img, label = train_data[sample_idx]
            #     figure.add_subplot(rows, cols, i)
            #     plt.title(labels_map[label])
            #     plt.axis("off")
            #     plt.imshow(to_pil(img))
            # plt.show()

        elif dataset_type == 2:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            
            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2])

        elif dataset_type == 3:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            
            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3])
        
        elif dataset_type == 4:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=augmented_transform, path_to_id=LABEL_FOLDER) 
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data4 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4])

        elif dataset_type == 5:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=augmented_transform, path_to_id=LABEL_FOLDER)
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data4 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            train_data5 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=augmented_transform, label=1)
            train_data6 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=augmented_transform, label=0)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6])

        elif dataset_type == 6:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            train_data4 = dataloaders.CatImagesWithLabels(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=data_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7])

            # to_pil = transforms.ToPILImage()
            # labels_map = {
            #     0: "Cat",
            #     1: "Dog"
            # }
            # figure = plt.figure(figsize=(16, 16))
            # cols, rows = 4, 4
            # for i in range(1, cols * rows + 1):
            #     sample_idx = torch.randint(len(train_data), size=(1,)).item()
            #     img, label = train_data[sample_idx]
            #     figure.add_subplot(rows, cols, i)
            #     plt.title(labels_map[label])
            #     plt.axis("off")
            #     plt.imshow(to_pil(img))
            # plt.show()

        elif dataset_type == 7:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            train_data4 = dataloaders.CatImagesWithLabels(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=data_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)
            train_data8 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=augmented_transform)
            train_data9 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=augmented_transform)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7, train_data8, train_data9])

        elif dataset_type == 8:
            # Per addestramento con label di ogni razza. Solo su dataset originale
            train_data = dataloaders.OriginalDatasetSpecies(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)

        elif dataset_type == 9:
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=data_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)

            train_data = torch.utils.data.ConcatDataset([train_data5, train_data6, train_data7])
        
        elif dataset_type == 10:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data2 = dataloaders.CustomDataset2(root=TRAIN_FOLDER2, transform=data_transform, label=1)
            train_data3 = dataloaders.CustomDataset2(root=TRAIN_FOLDER3, transform=data_transform, label=0)
            train_data4 = dataloaders.CatImagesWithLabels(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=augmented_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=augmented_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=augmented_transform)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7])

        
        elif dataset_type == 11:
            train_data1 = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data4 = dataloaders.CatImagesWithLabels(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data5 = dataloaders.DogImagesWithLabels(root_dir=TRAIN_FOLDER5, transform=data_transform)
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data7 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)

            train_data = torch.utils.data.ConcatDataset([train_data1, train_data4, train_data5, train_data6, train_data7])

        elif dataset_type == 12:
            train_data4 = dataloaders.CatFaces(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data = train_data4

        elif dataset_type == 13:
            train_data4 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)
            train_data = train_data4
        
        elif dataset_type == 14:
            train_data6 = dataloaders.CatImagesWithoutLabels(root_dir=TRAIN_FOLDER6, transform=data_transform)
            train_data = train_data6
        
        elif dataset_type == 15:
            train_data1 = dataloaders.CatFaces(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data2 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)
            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2])

        elif dataset_type == 16:
            train_data = dataloaders.CustomDataset2(root=TRAIN_FOLDER1, transform=data_transform, path_to_id=LABEL_FOLDER)
            train_data1 = dataloaders.CatFaces(root_dir=TRAIN_FOLDER4, transform=data_transform)
            train_data2 = dataloaders.DogFaces(root_dir=TRAIN_FOLDER7, transform=data_transform)
            train_data = torch.utils.data.ConcatDataset([train_data, train_data1, train_data2])

        elif dataset_type == 17:
            train_data = dataloaders.CatImagesWithLabels(root_dir=TRAIN_FOLDER4, transform=data_transform)

        dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=batch_size,
                                                shuffle=True)

        #Plot some training images
        # real_batch = next(iter(dataloader))
        # plt.figure(figsize=(16,16))
        # plt.axis("off")
        # plt.title("Training Images")
        # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, 
        #                                         normalize=True).cpu(),(1,2,0)))


    # Qui si fa il loading del dataset MNIST. Usato per controllare che il modello funzionase correttamente.
    elif dataset == 'MNIST':
        # MNIST dataset
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,))])

        mnist_data = torchvision.datasets.MNIST(root='./MNIST',
                                train=True,
                                transform=transform,
                                download=False)

        dataloader = torch.utils.data.DataLoader(dataset=mnist_data,
                                                batch_size=batch_size,
                                                shuffle=True)


    return dataloader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Funzione per la crezione del modello del generatore (conditional gan). In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_Cgen(model_path, use_pretrained_gen, nc, label_dim):

    # Create the generator
    if image_size == 32:
        netG = models.CGANGen32(nc, label_dim).to(device)
    elif image_size == 64:
        netG = models.CGANGen64(nc, label_dim).to(device)
    elif image_size == 128:
        netG = models.CGANGen128(nc, label_dim).to(device)
    elif image_size == 256:
        netG = models.CGANGen256(nc, label_dim).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init) 

    # Se specificato nel file json, si carica un modello pre addestrato.
    if use_pretrained_gen:
        print("PRE TRAINED")
        checkpoint = torch.load(model_path)
        for checkpoint_key, checkpoint_value in checkpoint.items():
            if checkpoint_key in netG.state_dict():
                netG.state_dict()[checkpoint_key].copy_(checkpoint_value)            

    # Print the model
    print(netG)

    return netG

# Funzione per la crezione del modello del discriminatore (conditional gan). In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_Cdis(nc, label_dim):

    # Create the Discriminator
    if image_size == 32:
        netD = models.CGANDis32(nc, label_dim).to(device)
    if image_size == 64:
        netD = models.CGANDis64(nc, label_dim).to(device)
    elif image_size == 128:
        netD = models.CGANDis128(nc, label_dim).to(device)
    elif image_size == 256:
        netD = models.CGANDis256(nc, label_dim).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    return netD

def filter_keys(dictionary, target_word):
    return{key: value for key, value in dictionary.items() if target_word in key}

# Funzione per la crezione del modello del generatore (weiestrass gan). In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_Wgen(model_path, use_pretrained_gen, use_pretrained_vae, nc, label_dim):

    # Create the generator
    if image_size == 32:
        netG = models.WGANGen32(nc, label_dim).to(device)
    elif image_size == 64:
        netG = models.WGANGen64(nc, label_dim).to(device)
    elif image_size == 128:
        netG = models.WGANGen128(nc, label_dim).to(device)
    elif image_size == 256:
        netG = models.WGANGen256(nc, label_dim).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    if not use_pretrained_gen:
        netG.apply(weights_init)

    # Anche nel caso della CGN, è possibile caricare un generatore pre addestrato.
    elif use_pretrained_gen:
        print("PRE TRAINED")
        checkpoint = torch.load(model_path)
        # print(checkpoint.keys())
        # for key, value in netG.state_dict().items():
        #     print("curr model", key)
        for checkpoint_key, checkpoint_value in checkpoint.items():
            if checkpoint_key in netG.state_dict():
                netG.state_dict()[checkpoint_key].copy_(checkpoint_value)

    # if use_pretrained_vae:  
    #     checkpoint = torch.load(model_path)

    #     target_word_checkpoint = 'Decoder'
    #     target_word_gen = 'main'
    #     print(checkpoint.keys())
    #     for key, value in netG.state_dict().items():
    #         print("curr model", key)
    #     filtered_keys_checkpoint = filter_keys(checkpoint, target_word_checkpoint)
    #     filtered_keys_gen = filter_keys(netG.state_dict(), target_word_gen)

    #     for checkpoint_key, checkpoint_value in filtered_keys_checkpoint.items():
    #         key_suffix = checkpoint_key.split(target_word_checkpoint)[-1]
    #         key_model = target_word_gen + key_suffix
    #         if key_model in netG.state_dict():
    #             netG.state_dict()[key_model].copy_(checkpoint_value) 

    # Print the model
    print(netG)

    return netG

# Funzione per la crezione del modello del discriminatore (weiestrass gan). In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_Wdis(nc, label_dim):

    # Create the Discriminator
    if image_size == 32:
        netD = models.WGANDis32(nc, label_dim).to(device)
    elif image_size == 64:
        netD = models.WGANDis64(nc, label_dim).to(device)
    elif image_size == 128:
        netD = models.WGANDis128(nc, label_dim).to(device)
    elif image_size == 256:
        netD = models.WGANDis256(nc, label_dim).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    return netD

# Funzione per la crezione del modello del generatore. In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_gen(nc, use_pretrained_vae, va_path):

    # Create the generator
    if image_size == 32:
        netG = models.Generator32(nc).to(device)
    elif image_size == 64:
        netG = models.Generator64(nc).to(device)
    elif image_size == 128:
        netG = models.Generator128(nc).to(device)
    elif image_size == 256:
        netG = models.Generator256(nc).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init) 

    if use_pretrained_vae:  
        # Carico il modello
        checkpoint = torch.load(va_path)

        target_word_checkpoint = 'Decoder'
        target_word_gen = 'main'
        print(checkpoint.keys())
        # for key, value in netG.state_dict().items():
        #     print("curr model", key)
        # Prendo solo i pesi relativi al decoder del VA
        filtered_keys_checkpoint = filter_keys(checkpoint, target_word_checkpoint)
        # filtered_keys_gen = filter_keys(netG.state_dict(), target_word_gen)

        for checkpoint_key, checkpoint_value in filtered_keys_checkpoint.items():
            # Sostituisco "main" a "Decoder" nel nome del layer, in modo da avere corrispondenza
            # di nomi tra GAN e VA 
            key_suffix = checkpoint_key.split(target_word_checkpoint)[-1]
            key_model = target_word_gen + key_suffix
            print("key_model ", key_model)
            # Carico i pesi dal decoder del VA al generatore della GAN
            if key_model in netG.state_dict():
                netG.state_dict()[key_model].copy_(checkpoint_value)                    

    # Print the model
    print(netG)

    return netG

# Funzione per la crezione del modello del discriminatore. In base alla grandezza delle 
# immagini desiderata, si crea il modello appropriato.
def create_dis(nc):

    # Create the Discriminator
    if image_size == 32:
        netD = models.Discriminator32(nc).to(device)
    elif image_size == 64:
        netD = models.Discriminator64(nc).to(device)
    elif image_size == 128:
        netD = models.Discriminator128(nc).to(device)
    elif image_size == 256:
        netD = models.Discriminator256(nc).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    return netD


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def label_preprocess():
    # label preprocessing
    # for the generator we will use onehot vectors
    # for the discriminator we need onehot "images"
    onehot = torch.zeros(label_dim, label_dim)

    if label_dim == 2:
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(label_dim, 1), 1)    #for animals (cat or dog - no breed)

    #for animals (class id -> instead of condition on the breed, we condition on the class (Abyssininan, bulldog ecc..))
    elif label_dim == 37:
        shorter_tensor = torch.arange(37, dtype=torch.long)
        onehot = onehot.scatter_(1, shorter_tensor.view(label_dim, 1), 1)    
    else:
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1)     #for mnist
    print(onehot)
    onehot = onehot.view(label_dim, label_dim, 1, 1)

    fill = torch.zeros([label_dim, label_dim, image_size, image_size])
    for i in range(label_dim):
        fill[i, i, :, :] = 1

    return fill, onehot

# Metodo per definire loss e ottimizzatore della rete. In base al file di configurazione
# il codice inizializza loss e ottimizzatori specifici.
def initializion(netG, netD, loss_type, optimizer):
    # Initialize BCELoss function

    if loss_type == "bce":
        criterion = nn.BCELoss()
    elif loss_type == "L1":
        criterion = nn.L1Loss()
    elif loss_type == "MSE":
        criterion = nn.MSELoss()
    elif loss_type == "SmoothL1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "Hinge":
        # NOTA BENE: La Hinge loss si aspetta 1 e -1 come label, anzichè 0 1.
        criterion = nn.HingeEmbeddingLoss()
    #criterion = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(test_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    if loss_type == "Hinge":
        real_label = 1.
        fake_label = -1.


    if optimizer == "adam":
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=dis_learning_rate, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=gen_learning_rate, betas=(beta1, 0.999))
    elif optimizer == "adamaX":
        optimizerD = optim.Adamax(netD.parameters(), lr=dis_learning_rate, betas=(beta1, 0.999))
        optimizerG = optim.Adamax(netG.parameters(), lr=gen_learning_rate, betas=(beta1, 0.999))
    elif optimizer == "sgd":
        optimizerD = optim.SGD(netD.parameters(), lr=dis_learning_rate, momentum=momentum)
        optimizerG = optim.SGD(netG.parameters(), lr=gen_learning_rate, momentum=momentum)
    elif optimizer == "rmsprop":
        # Con rmsprop meglio un learning rate basso
        optimizerD = optim.RMSprop(netD.parameters(), lr=dis_learning_rate, alpha = 0.9)
        optimizerG = optim.RMSprop(netG.parameters(), lr=gen_learning_rate, alpha = 0.9)

    return criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD

# function to generate test sample
def generate_test(fixed_noise, onehot, G):
    G.eval()
    # label 0
    c = (torch.ones(test_size)*0).type(torch.LongTensor)
    c_onehot = onehot[c].to(device)
    out = G(fixed_noise, c_onehot)
    inference_res = out

    # labels 1-label_dim (10 for mnist, 2 for animals)
    for l in range(1,label_dim):
        c = (torch.ones(test_size)*l).type(torch.LongTensor)
        c_onehot = onehot[c].to(device)
        out = G(fixed_noise, c_onehot)
        inference_res = torch.cat([inference_res, out], dim = 0)
    G.train()
    return inference_res

# Metodo che contiene il loop di addestramento della cgan
def train_cgan(netG, netD, dataloader, criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD, fill, onehot):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (imgs, labels) in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            b_size = imgs.size(0)
            
            # Establish convention for real and fake labels during training
            # Let's do it more simply than last time
            real_label = torch.ones(b_size).to(device)
            fake_label = torch.zeros(b_size).to(device)
            
            # Format batch
            real_cpu = imgs.to(device)
            c_fill = fill[labels].to(device)
            # Forward pass real batch through D
            output = netD(real_cpu, c_fill).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # pick random labels ang generate corresponding onehot
            c = (torch.rand(b_size) * label_dim).type(torch.LongTensor) #equivalent to int64 #[0,6,4,3,9]
            c_onehot = onehot[c].to(device)
            # Generate fake image batch with G
            fake = netG(noise, c_onehot)
            # Classify all fake batch with D
            c_fill = fill[c].to(device)
            output = netD(fake.detach(), c_fill).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, c_fill).view(-1)
            # Calculate G's loss based on this output 
            errG = criterion(output, real_label) # fake images are real for generator cost
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 400 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generate_test(fixed_noise, onehot, netG).detach().cpu()
                im_grid = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(im_grid)
                vutils.save_image(im_grid, os.path.join(PATH_TO_SAVE, "Grid images/{}_{}.jpg").format(epoch, iters))

            # # save 500 images generated from random noise in order to calculate FID score
            # if(epoch == num_epochs -1 and (iters % 250 == 0)):
            #     with torch.no_grad():
            #         fake = generate_test(fixed_noise, onehot, netG).detach().cpu()
            #     for item in range(0, test_size):
            #         vutils.save_image(fake[item], os.path.join(PATH_TO_SAVE, "Fake images/{}.jpg").format(item))

            iters += 1
        # Applica il decadimento del learning rate a fine di ogni epoca
        scheduler_d.step()
        scheduler_g.step()
        torch.save(netG.state_dict(), os.path.join(PATH_TO_SAVE, "netG_conditional.pth"))
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH_TO_SAVE, "losses.png"))
    plt.close()


def train_wgan(netG, netD, dataloader, fixed_noise, optimizerG, optimizerD, fill, onehot):

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    print("Starting Training Loop...")

    for epoch in range(num_epochs): 
        # For each batch in the dataloader
        for i, (imgs, labels) in enumerate(dataloader, 0):
            
            if imgs.size(0) != batch_size:
                continue
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            b_size = imgs.size(0)
            
            # NO NEED FOR LABELS IN WGAN
            # real_label = torch.ones(b_size).to(device)
            # fake_label = torch.zeros(b_size).to(device)
            
            # Format batch
            real_cpu = imgs.to(device)
            c_fill = fill[labels].to(device)
            # Forward pass real batch through D
            output = netD(real_cpu, c_fill).view(-1)
            # Calculate loss on all-real batch
            errD_real = - torch.mean(output)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            c_onehot = onehot[labels].to(device)
            # Generate fake image batch with G
            fake = netG(noise, c_onehot)
            # Classify all fake batch with D
            c_fill = fill[labels].to(device)
            output = netD(fake.detach(), c_fill).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = torch.mean(output)
            
            # caluculate gradient penalty
            alpha = torch.rand(real_cpu.size(0), 1, 1, 1).to(device) # alpha*x + (1 - alpha)*x_2
            x_hat = (alpha * real_cpu.data + (1 - alpha) * fake.data).requires_grad_(True)
            output = netD(x_hat, c_fill)
            d_loss_gp = gradient_penalty(output, x_hat)
            
            # backpropagate loss
            totalD_loss = errD_real + errD_fake + 10*d_loss_gp
            totalD_loss.backward()
            
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, c_fill).view(-1)
            # Calculate G's loss based on this output
            errG = - torch.mean(output)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 400 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)): 
                
                with torch.no_grad():
                    fake = generate_test(fixed_noise, onehot, netG).detach().cpu()
                im_grid = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(im_grid)
                vutils.save_image(im_grid, os.path.join(PATH_TO_SAVE, "Grid images/{}_{}.jpg").format(epoch, iters))
            
            # # save 500 images generated from random noise in order to calculate FID score
            # if(epoch == num_epochs -1 and (i >= len(dataloader)-10)):
            #     with torch.no_grad():
            #         fake = generate_test(fixed_noise, onehot, netG).detach().cpu()
            #     for item in range(0, test_size):
            #         vutils.save_image(fake[item], os.path.join(PATH_TO_SAVE, "Fake images/{}.jpg").format(item))

            iters += 1
        # Applica il decadimento del learning rate a fine di ogni epoca
        scheduler_d.step()
        scheduler_g.step()
        torch.save(netG.state_dict(), os.path.join(PATH_TO_SAVE, "netWG.pth"))
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH_TO_SAVE, "losses.png"))
    plt.close()

def train_VA(vae, train_loader, bce_loss, optimizer):

    vae.train()

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # send to device
            data = data.to(device)
            
            optimizer.zero_grad()
            output, log_var, mean = vae(data)

            bce = bce_loss(output, data)
            kl = vae.KL_loss(log_var, mean)
            loss = bce + kl
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE: {:.6f} KL: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), bce.item(), kl.item()))
                
                out_grid = torchvision.utils.make_grid(output[:8])
                torchvision.utils.save_image(out_grid, os.path.join(PATH_TO_SAVE, "Grid images/{}_{}.jpg").format(epoch, batch_idx))
        
        # save model every epoch
        torch.save(vae.state_dict(), os.path.join(PATH_TO_SAVE, "vae.pth"))


def test_VA(vae, test_loader):
    vae.load_state_dict(torch.load(os.path.join(PATH_TO_SAVE, "vae.pth")))

    # TEST
    image, target = next(iter(test_loader)) # load one batch
    image = image.to(device)

    vae.eval()

    out, _, _ = vae(image)

    # input
    plt.figure()
    to_pil = transforms.ToPILImage()
    image_grid = torchvision.utils.make_grid(image[:8])
    plt.imshow(to_pil(image_grid))
    plt.axis("off")
    plt.show()

    # output
    plt.figure()
    out_grid = torchvision.utils.make_grid(out[:8])
    torchvision.utils.save_image(out_grid, os.path.join(PATH_TO_SAVE, "Fake images/test1.jpg"))
    plt.imshow(to_pil(out_grid))
    plt.axis("off")
    plt.show()

    # GENERATE!!!!
    z = torch.randn(8,2).to(device)
    print(z.size())

    generated_samples = vae.generate_img(z)

    plt.figure()
    gen_grid = torchvision.utils.make_grid(generated_samples)
    torchvision.utils.save_image(gen_grid, os.path.join(PATH_TO_SAVE, "Fake images/test2.jpg"))
    # plt.imshow(to_pil(gen_grid))
    # plt.axis("off")
    # plt.show()

# Questo metodo contiene il loop per addestrare la GAN classica.
def train_model(netG, netD, dataloader, criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD):
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Scheduler per diminuire il learning rate ad ogni epoca.
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch # Make accumalated gradients of the discriminator zero.
            netD.zero_grad()
            # Format batch
            # Transfer data tensor to GPU/CPU (device)
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # Create labels for the real data. (label=1)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Calculate the output of the discriminator of the fake data.
            # As no gradients w.r.t. the generator parameters are to be
            # calculated, detach() is used. Hence, only gradients w.r.t. the
            # discriminator parameters will be calculated.
            # This is done because the loss functions for the discriminator
            # and the generator are slightly different.
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            #skip_disc = 2
            # if epoch > 70:
            #     skip_disc = 4

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 15 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i  % 600 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                im_grid = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(im_grid)
                vutils.save_image(im_grid, os.path.join(PATH_TO_SAVE, "Grid images/{}_{}.jpg").format(epoch, iters))

            # # save 500 images generated from random noise in order to calculate FID score
            # if(epoch == num_epochs -1 and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         for item in range(0, test_size):
            #             fake = netG(fixed_noise).detach().cpu()
            #             vutils.save_image(fake[item], os.path.join(PATH_TO_SAVE, "Fake images/{}.jpg").format(item))

            iters += 1

        # Salvo i dati utili su tensorboard
        # writer.add_scalar("gen error", errG.item(), epoch)
        # writer.add_scalar("dis error", errD.item(), epoch)
        # writer.add_image("grid image", im_grid, epoch)
            
        # Applico lo scheduler all'ottimizzatore, riducendo il leraning rate
        # scheduler_d.step()
        # scheduler_g.step()

        torch.save(netG.state_dict(), os.path.join(PATH_TO_SAVE, "netG.pth"))

    #save the loss plotting on computer 
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH_TO_SAVE, "losses.png"))
    plt.close()
    #plt.show()


def main():

    global image_size, batch_size, beta1, num_epochs, nc, gen_learning_rate, dis_learning_rate, momentum, label_dim, test_size

    # Lettura dei prametri da file json
    with open("config.json", 'r') as json_file:
        config = json.load(json_file)
        model_type = config.get('model_type', 'GAN')
        image_size = config.get('image_size', 64)
        batch_size = config.get('batch_size', 128)
        nc = config.get('num_channels', 3)
        num_epochs = config.get('num_epochs', 100)
        gen_learning_rate = config.get('gen_learning_rate', 0.0002)
        dis_learning_rate = config.get('dis_learning_rate', 0.0002)
        vae_lr = config.get('vae_lr', 0.001)
        beta1 = config.get('beta1', 0.5)
        momentum = config.get('momentum', 0.9)
        loss_type = config.get('loss_type', 'mse')
        optimizer = config.get('optimizer', 'adam')
        dataset_type = config.get('dataset_type', 1)
        dataset = config.get('dataset', 'ANIMALS')
        label_dim = config.get('label_dim', 2)
        test_size = config.get('test_size', 32)
        use_pretrained_gen = config.get('use_pretrained_gen', False)
        model_path = config.get('model_path', './Models')
        use_pretrained_vae = config.get('use_pretrained_vae', False)
        va_path = config.get('va_path', './Models')

    # Una serie di condizioni per controllare qualche modello è tato specificato nel file di configurazione
    # per poter lanciare l'addestramento giusto
    if model_type == "GAN":
        set_seed()
        dataloader = data_loading(dataset_type, dataset=dataset)

        netG = create_gen(nc, use_pretrained_vae, va_path)
        netD = create_dis(nc)

        criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD = initializion(netG, netD, loss_type, optimizer)
        
        train_model(netG, netD, dataloader, criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD)

    if model_type == "VA":
        if image_size == 32:
            vae = models.VAE(image_channels=nc).to(device)
        else:
            # vae = models.VAE64(image_channels=nc).to(device)
            vae = models.VAE64WithBN(image_channels=nc).to(device)
        train_loader, test_loader = data_loading_VA(dataset_type, dataset=dataset)
        criterion = nn.BCELoss(reduction="sum")

        # In base alla scelta effettuata nel file json, viene inizializzato l'ottimizzatore appropriato per
        # il VA
        if optimizer == "adam":
            # Setup Adam optimizers for both G and D
            optimizerVA = optim.Adam(vae.parameters(), lr=vae_lr, betas=(beta1, 0.999))
        elif optimizer == "adamaX":
            optimizerVA = optim.Adamax(vae.parameters(), lr=vae_lr, betas=(beta1, 0.999))
        elif optimizer == "sgd":
            optimizerVA = optim.SGD(vae.parameters(), lr=vae_lr, momentum=momentum)
        elif optimizer == "rmsprop":
            # Con rmsprop meglio un learning rate basso
            optimizerVA = optim.RMSprop(vae.parameters(), lr=vae_lr, alpha = 0.9)

        train_VA(vae, train_loader, criterion, optimizerVA)
        test_VA(vae, test_loader)

    if model_type == "WGAN":
        set_seed()
        dataloader = data_loading(dataset_type, dataset=dataset)

        netG = create_Wgen(model_path, use_pretrained_gen, use_pretrained_vae, nc, label_dim)
        netD = create_Wdis(nc, label_dim)

        criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD = initializion(netG, netD, loss_type, optimizer)
        fill, onehot = label_preprocess()

        train_wgan(netG, netD, dataloader, fixed_noise, optimizerG, optimizerD, fill, onehot)

    if model_type == "CGAN":
        set_seed()
        dataloader = data_loading(dataset_type, dataset=dataset)

        netG = create_Cgen(model_path, use_pretrained_gen, nc, label_dim)
        netD = create_Cdis(nc, label_dim)

        criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD = initializion(netG, netD, loss_type, optimizer)
        fill, onehot = label_preprocess()

        train_cgan(netG, netD, dataloader, criterion, fixed_noise, real_label, fake_label, optimizerG, optimizerD, fill, onehot)


if __name__ == "__main__":
    main()