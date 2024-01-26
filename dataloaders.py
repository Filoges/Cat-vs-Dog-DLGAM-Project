import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import cv2
from torchvision.io import read_image

# Prima versione del dataloader del dataset originale
# utilizza openCV e si aspetta le trasformate di Albumentation
class CustomDataset(Dataset):
    def __init__(self, root, transform, label=2, path_to_id=""):
        self.path = root
        self.transf = transform
        self.label = label
        self.species_map = {}

        self.file_list = [os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]  

        if label == 2 and path_to_id != "":
            # Read the txt file to create the species_map
            with open(path_to_id, 'r') as file:
                lines = file.readlines()
                for line in lines[6:]:  # Skipping the first 5 lines with comments
                    parts = line.strip().split()
                    image_name_parts = parts[0].split("_")
                    image_name = image_name_parts[0]
                    species = int(parts[2]) - 1  # Assuming the species is the third item after splitting
                    self.species_map[image_name] = species


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):

        if self.label == 2:
            img = cv2.imread(self.file_list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented = self.transf(image = img)
            transformed_img = augmented['image']

            image_name_parts = os.path.splitext(os.path.basename(self.file_list[i]))[0].split("_")
            image_name = image_name_parts[0]
            label = self.species_map[image_name]

        else:
            img = cv2.imread(self.file_list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented = self.transf(image=img)
            transformed_img = augmented['image']

            label = self.label
        
        return transformed_img, label
    

# Seconda versione del dataloader per il dataset originale.
# Non utilizza openCV e si aspetta le trasformate di pyTorch
# Può essere usato per altri dataset dove tutte le immagini sono in una singola cartella e 
# si conosce la label delle immagini.
class CustomDataset2(Dataset):
    # A questo dataloader è possibile passare una label specifica (0 o 1) se si sa già la classe di appartenenza
    # delle immagini (gatto o cane). Questo dataloader carica tutte le immagini dentro una cartella.
    # Se label=2 siamo in presenza del dataset originale, dunque per ogni immagine si va a prendere la 
    # classe di appartenenza da un file specifico nella cartella del dataset.
    # Se label viene passato come valore pari a 0 o 1, quel valore sarà associato a tutte le immagini della
    # cartella.
    def __init__(self, root, transform, label=2, path_to_id=""):
        self.path = root
        self.transf = transform
        self.label = label
        self.species_map = {}

        self.file_list = [os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]  

        if label == 2 and path_to_id != "":
            # Read the txt file to create the species_map
            with open(path_to_id, 'r') as file:
                lines = file.readlines()
                for line in lines[6:]:  # Skipping the first 5 lines with comments
                    parts = line.strip().split()
                    image_name_parts = parts[0].split("_")
                    image_name = image_name_parts[0]
                    # ci tolgo 1 perché nel file la label è: 1 gatto, 2 cane. Io voglio 0 gatto, 1 cane
                    species = int(parts[2]) - 1  # Assuming the species is the third item after splitting
                    self.species_map[image_name] = species


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):

        if self.label == 2:
            img = Image.open(self.file_list[i])
            img = img.convert('RGB')
            #img = read_image(self.file_list[i])
            transformed_img = self.transf(img)

            image_name_parts = os.path.splitext(os.path.basename(self.file_list[i]))[0].split("_")
            image_name = image_name_parts[0]
            label = self.species_map[image_name]

        else:
            img = Image.open(self.file_list[i])
            img = img.convert('RGB')
            #img =  read_image(self.file_list[i])
            transformed_img = self.transf(img)

            label = self.label
        
        return transformed_img, label


# Questo dataset è specifico per il dataset originale fornito con le specifiche del progetto. Questo perché restituice una label 
# compresa tra 0 e 36, ovvero il numero di specie diverse presenti nel dataset (pitbull, ragdoll ecc...). Essendo che per forza
# ha bisogno del file con le informazioni su specie e razze, può essere usato solo con il dataset originale.  
class OriginalDatasetSpecies(Dataset):
    def __init__(self, root, transform, path_to_id):
        self.path = root
        self.transf = transform
        self.label = 0
        self.species_map = {}

        self.file_list = [os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]  

        # Read the txt file to create the species_map
        with open(path_to_id, 'r') as file:
            lines = file.readlines()
            for line in lines[6:]:  # Skipping the first 5 lines with comments
                parts = line.strip().split()
                image_name_parts = parts[0].split("_")
                image_name = image_name_parts[0]
                # ci tolgo 1 perché nel file la label è: 1 gatto, 2 cane. Io voglio 0 gatto, 1 cane
                class_id = int(parts[1]) - 1  # Assuming the species is the third item after splitting
                self.species_map[image_name] = class_id


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):

        img = Image.open(self.file_list[i])
        img = img.convert('RGB')
        #img = read_image(self.file_list[i])
        transformed_img = self.transf(img)

        image_name_parts = os.path.splitext(os.path.basename(self.file_list[i]))[0].split("_")
        image_name = image_name_parts[0]
        label = self.species_map[image_name]
        
        return transformed_img, label
    
# Questo dataloader guarda tutte le sottocartelle presenti nel dataset e prende tutte le immagini.
# Si tratta di un dataset composto da solo cani.
class DogImagesWithLabels(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transf = transform

        for breed_folder in os.listdir(root_dir):
            breed_path = os.path.join(root_dir, breed_folder)
            if os.path.isdir(breed_path):
                for image_name in os.listdir(breed_path):
                    image_path = os.path.join(breed_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(breed_folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        transformed_img = self.transf(image)

        # 0 for cats, 1 for dogs
        return transformed_img, 1
    
# Questo dataloader prende in ingresso lo stesso datatset del dataloader "Cat faces"
# tuttavia prendendo tutte le immagini di tutte le sottocartelle.
class CatImagesWithLabels(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transf = transform

        for subset in os.listdir(root_dir):
            subset_path = os.path.join(root_dir, subset)
            if os.path.isdir(subset_path):
                for breed_folder in os.listdir(subset_path):
                    breed_path = os.path.join(subset_path, breed_folder)
                    if os.path.isdir(breed_path):
                        for image_name in os.listdir(breed_path):
                            image_path = os.path.join(breed_path, image_name)
                            self.image_paths.append(image_path)
                            self.labels.append(breed_folder.split()[-1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        transformed_img = self.transf(image)

        return transformed_img, 0

# Un generico dataset di gatti caricato per cercare di ovviare al problema del numero di immagini.
# In questo caso il dataloader entra nelle tre cartelle di trin, validatione e test.
class CatImagesWithoutLabels(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = []
        self.transf = transform

        subset_folders = ["train", "test", "validation"]

        for subset in subset_folders:
            subset_path = os.path.join(root_dir, subset)
            if os.path.isdir(subset_path):
                for image_name in os.listdir(subset_path):
                    if image_name.endswith(".jpg"):
                        image_path = os.path.join(subset_path, image_name)
                        self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transf(image)

        return transformed_image, 0
    
# Questo dataloader crica le immagini di un altro dataet scaricato. Un dataset composto da sole immagini 
# di facce di cani
class DogFaces(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = []
        self.transf = transform

        subset_folders = ["train", "test", "validation"]

        for subset in subset_folders:
            subset_path = os.path.join(root_dir, subset)
            if os.path.isdir(subset_path):
                for image_name in os.listdir(subset_path):
                    if image_name.endswith(".jpg"):
                        image_path = os.path.join(subset_path, image_name)
                        self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transf(image)

        return transformed_image, 1
    
# Questo dataloader serve per caricare le immagini di uno specifico subset di immagini
# all'interno di un dataset più ampio. In particolare si tratta di sole immagini di facce
# di gatti.
class CatFaces(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = []
        self.transf = transform

        for subset in os.listdir(root_dir):
            subset_path = os.path.join(root_dir, subset)
            if os.path.isdir(subset_path):
                for breed_folder in os.listdir(subset_path):
                    if breed_folder == "animal animal_faces cat cat_face":
                        breed_path = os.path.join(subset_path, breed_folder)
                        if os.path.isdir(breed_path):
                            for image_name in os.listdir(breed_path):
                                image_path = os.path.join(breed_path, image_name)
                                self.image_paths.append(image_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')
        transformed_img = self.transf(image)

        return transformed_img, 0
        

class IIITDataset(Dataset):
    def __init__(self, df, tfm=None):
        self.df = df
        self.tfm = tfm
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        img = Image.open(self.df.image.iloc[i]).convert('RGB')
        mask = Image.open(self.df.trimap.iloc[i])
        img = np.asarray(img)
        mask = np.asarray(mask)
        if self.tfm:
            augmented = self.tfm(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return img, mask
        

def create_dataset(image_size):
    
    df = pd.read_csv(
        "oxford-iiit-pet/annotations/list.txt",
        delimiter=" ",
        skiprows=6,
        header=None,
        names=["stem", "class_id", "species", "breed"]
    )
    df["class_name"] = df.stem.map(lambda x: x.split("_")[0])
    df["image"] = df.stem.map(lambda x: f"oxford-iiit-pet/images/{x}.jpg")
    df["trimap"] = df.stem.map(lambda x: f"oxford-iiit-pet/annotations/trimaps/{x}.png")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tfm = A.Compose([
        A.Resize(image_size, image_size,  interpolation=1, always_apply=True, p=1),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean = imagenet_mean, std = imagenet_std),
        ToTensorV2(),
    ])

    val_tfm = A.Compose([
        A.Resize(image_size, image_size,  interpolation=1, always_apply=True, p=1),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean = imagenet_mean, std = imagenet_std),
        ToTensorV2(),
    ])

    augm_train = A.Compose([
        A.Resize(image_size, image_size,  interpolation=1, always_apply=True, p=1),
        A.CenterCrop(image_size, image_size),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=0.5), 
        A.RandomRotate90(),
        A.PixelDropout (dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=True, p=0.5),
        A.Normalize(mean = imagenet_mean, std = imagenet_std),
        ToTensorV2(),
    ])

    skf = StratifiedKFold(5)
    #train_idx, val_idx = next(iter(skf.split(df, df.class_id)))
    train_df = df.iloc[:]
    augm_df = df.iloc[:]
    #val_df = df.iloc[val_idx]

    train_ds = IIITDataset(train_df, tfm=train_tfm)
    augm_ds = IIITDataset(augm_df, tfm=augm_train)
    train_ = torch.utils.data.ConcatDataset([train_ds, augm_ds])
    #val_ds = IIITDataset(val_df, tfm=val_tfm)

    return train_



