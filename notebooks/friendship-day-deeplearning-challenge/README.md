Donwload the required Dataset from -> https://www.kaggle.com/bing101/friendshipgoals

First of all: Run the data_preocessing.ipynb files to create features and labels .npy files

I have written several models for this challenge and given them names accordingly.

There are two models that worked the best:
The code for these models are :-

- Resnet34

You will find this model in the file resnet34And50.ipynb
you will need to install FAST.ai liberary to use this

- VGG16 
Another model is in the vgg16-experiment.ipynb
Named: VGG16 all frozen layers and Additional Conv layers

This model is trained on tensorflow


## Approach:-

First I collected the dataset for my model. I used websites such as Shutter stock and google images.
I have near about 1000 samples for each label class.

I have used transfer learning for building my model. 
I used vgg16 and resnet34 and 50.

Vgg16 is trained on tensorflow. I tried different approaches such as 

- freezing all the layers, saving weights, and loading those weights in fresh vgg16 and training again.

- Adding some convulutional layer to the end

-- Unfreezing last two layers 

-- Freezing entire model and adding two dense layers

Resnet 34 and 50 are trained on Fast.ai lib. For this:
- Used the same freezing, unfreezing approach 
- I plotted the lr_finder to determine learning rate


