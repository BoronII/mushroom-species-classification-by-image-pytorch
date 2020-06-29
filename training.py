# Mushroom  Species Classification 
# The purpose of this project is for me to learn about deep learning with PyTorch, using the fastai front end. 
# **Keywords:** neural nets, deep learning, pytorch, fastai, transfer learning, fine tuning, one cycle learning, boot strapping, fine grain classification. 

# Being able to tell one mushroom species from another can mean the difference between picking yourself a tasty meal or ending up with something that could make you sick or even lead to death. Approximately 14,000 species of mushrooms have been cataloged world wide and creating a classifier that has been trained on all of these species presents a data collection challenge that I was not prepared to undertake. In this project I wanted to perform the data collection and data cleaning steps myself, to keep things managable I decided to work toward creating a classifier that will be useful to people foraging in a more localized geographical area. I learned that mushroom foraging is a popular activity in the Canadian province of British Columbia. Being a canadian myseld, this is the geographical region I decided to focus my efforts on. 

# Our data set contains an average of 229 images per category (max=*, min=*), this is more images than 
# - mention ratio between minority and majority classes since accuracy is used as the evaluation metric.

# I simply used the default set of transform for this initials version of the classifier. A good source for a more detailed explanation of these transforms with examples, is the fastai documentation for [get_transforms](https://docs.fast.ai/vision.transform.html#get_transforms). 

# This model was trained using the fastai implementation of the one cycle fitting policy developed by Leslie Smith. Kostas Mavropalias has laid out an excellent explaination of the one cycle fitting policy that can be found [here](https://iconof.com/1cycle-learning-rate-policy/)

#--------------------------------------------------------------------------------------

# We import all the necessary packages. We will be building our classifier using the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). 


from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import pandas as pd


# Batch size for SDG
bs = 64

# Creates a path object
path = Path('mushroom_classification_data_cleaned/')


df = pd.read_csv(path/'cleaned.csv', header='infer')

# talk about resizing
# talk about data augmentation here
# talk about spliting and the random seed
# talk about normalization
tfms = get_transforms()
np.random.seed(33)
data = ImageDataBunch.from_df(path, df, valid_pct=0.2, ds_tfms=tfms, size=224).normalize(imagenet_stats)


# Shows a random subset of the images. Doing this to see that data is not cropped or resized poorly.
data.show_batch(rows=5, figsize=(11,8))


# explain here what resnet50 is, and the idea of transfer learning.
learn = cnn_learner(data, models.resnet50, metrics=error_rate)

#
learn.lr_find()

#
learn.recorder.plot()

# talk about one cycle learning
learn.fit_one_cycle(10, max_lr=1e-2)

#
learn.save('stage-1')


# the loss function is cross entropy loss: did you predict the right label and how confidently? 
# say what it plots
learn.recorder.plot_losses()


learn.lr_find()
learn.recorder.plot()

# talk about how ive only trained last layer and what unfreeze does.
learn.unfreeze()

# talk about why not teach all wieghts at the same rate and what slice does here.
learn.fit_one_cycle(10, max_lr=slice(1e-5,5e-3))

learn.save('stage-2')

learn.recorder.plot_losses()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5))
learn.recorder.plot_losses()
learn.save('stage-final')

# The classificationInterpretation object contains all the information needed to interperate the model.
# Basically exactly the things that I will looking at in the next few cells
interp = ClassificationInterpretation.from_learner(learn)

# explain top losses and mention how it was used in the second data cleaning step.
interp.plot_top_losses(9, figsize=(20,13))

#
interp.plot_confusion_matrix(figsize=(20,20), dpi=120)

# Same info as in confusion matrix but easier to see the worst cases.
interp.most_confused(min_val=5)


