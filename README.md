# Cat-vs-Dog-DLGAM-Project

This file stores basic info on the project's files and how to use them.

## config.json
This is the configuration file for the project. Instead of writing a program that accepts command line arguments, I preferred to design the project in order to read a file, retrieve the necessary information, and then start the training. Next is an explanation of every argument in the file:
- "model_type" (str):  this parameter lets you choose between the available models (GAN, CGAN, WGAN, VA).
- "image_size" (int):  as the name suggests, using this parameter you can choose the image size you want to train on (32, 64, 128, 256).
- "batch_size" (int):  batch size value.
- "num_channels" (int):  number of channels that the model expects. Since this network was testes on MNIST too, there was the need to specify 1 channel tensors.  
- "num_epochs" (int):  number of epoches required for the training.
- "gen_learning_rate" (float):  learning rate for the generator class.
- "dis_learning_rate" (float):  learning rate for the discriminator class.
- "beta1" (float):  parameter for adam optimizer.
- "momentum" (float):  parameter for sgd optimizer.
- "loss_type" (str):  desired loss function (bce, L1, MSE, SmoothL1).
- "optimizer" (str):  desired optimizer (adam, adamaX, sgd, rmsprop).
- "dataset_type" (int):  since in this project were used many datasets, with this param it is possible to choose what configuration of datasets to use. Further explanation in the section below.
- "dataset" (str):  this network has been designed to work with animal images, but has been checked on MNIST dataset too. So with this param it is possible to choose the type of dataset desired (ANIMALS, MNIST).
- "label_dim" (int):  this param controls the size of the label vector. For the cat and dog dataset it's usually 2 (cat or dog), while for MNIST is 10 (the 10 different digits). It is possible, using only the original dataset (dataset_type = 8), to set this value to 37, to use race labels instead of species (ragdoll, german shepard, ecc...).
- "test_size" (int):  how many images will be used to make the grid view during training.
- "use_pretrained_gen" (bool):  it is possible to load a pretrained generator model, by setting this parameter to "True".
- "model_path" (str):  this path refers to the pretrained genrator .pth file.
- "use_pretrained_vae" (bool): this parameter is read by the network, but does nothing. In future updates of the code, a pretrained decoder of the VA can be used as generator for the training.

###dataset_type param
In the main.py file there is a method called "data_loading" that return the dataloader used later during training. In this method the "dataset_type" param is read, and a different dataloader is used based on its value.
For this project 4 different datasets hava been used:
- The cat and dogs stanford dataset (original dataset for this project)
- The dog faces dataset (https://images.cv/dataset/dog-face-image-classification-dataset)

## main.py
As the name suggests this is the main file, the file that launches the desired training. 
