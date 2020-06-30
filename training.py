# Mushroom  Species Classification 
# We import all the necessary packages. We will be building our classifier 
# Using the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which 
# Sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163).
 
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import error_rate

def find_and_plot_lr():
    '''Explores various learning rates. Creates a plot of learning rate vs loss 
    from the .lr_find() exploration. I used this, together with experimentation to 
    settle on good learning rates.''' 
    learn.lr_find()
    learn.recorder.plot()


    

if __name__=='__main__': 

    # Batch size for SGD.
    bs = 64
    path = Path('mushroom_classification_data_cleaned/')
    
    # Read the cleaned data into a pandas data frame.
    df = pd.read_csv(path/'cleaned.csv', header='infer')
    
    # I simply used the default set of transform for this initials version of the 
    # classifier. A good source for a more detailed explanation of these transforms 
    # with examples, is the fastai documentation for get_transforms() 
    # (https://docs.fast.ai/vision.transform.html#get_transforms). 
    tfms = get_transforms()
    
    # Set random seed so that the validation set will be the same each time
    # the script is run. 
    np.random.seed(33)
    
    # data is organized into a fastai databunch, 20% of the data is split to the 
    # validation set (refered to as the test set in sklearn).
    # The images are cropped and resized to 224 pixels by 224 pixels.
    data = ImageDataBunch.from_df(path, df, valid_pct=0.2, ds_tfms=tfms, size=224)
    
    # The data is normalized using the same stats the were used for pretraining.
    data = data.normalize(imagenet_stats)
    
    # Shows a random subset of the images. Doing this to see that data is not 
    # cropped or resized poorly. 
    data.show_batch(rows=5, figsize=(11,8))
    
    # Creates a learner object using the data bunch, together with the resnet50 
    # architecture and pretrained weights and biases. 
    learn = cnn_learner(data, models.resnet50, metrics=error_rate)
    
    find_and_plot_lr()
    
    # This model was trained using the fastai implementation of the one cycle 
    # fitting policy developed by Leslie Smith. Kostas Mavropalias has laid out an 
    # excellent explaination of the one cycle fitting policy that can be found 
    # here (https://iconof.com/1cycle-learning-rate-policy/)
    learn.fit_one_cycle(10, max_lr=1e-2)
    
    learn.save('stage-1')
    
    
    # the loss function is cross entropy loss.
    learn.recorder.plot_losses()
    
    find_and_plot_lr()
    
    
    # Up until now all the weights and biases from the pretrained model have been
    # held frozen. Only the output layer (which was replaced with an output layer
    # of size 73) has been updated by gradient descent. 
    # learn.unfreeze() allows all the parameters of the neural net to be updated.
    learn.unfreeze()
    
    # Here slice allows us to train the parameters (weights and biases) of early layers
    # more slowly than we train parameters coming from later layers of the model.
    # More information about why we want to do it this way here 
    # https://arxiv.org/pdf/1311.2901.pdf
    learn.fit_one_cycle(10, max_lr=slice(1e-5,5e-3))
    learn.save('stage-2')
    
    learn.recorder.plot_losses()
    find_and_plot_lr()
    
    learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5))
    learn.recorder.plot_losses()
    learn.save('stage-final')
    
    # The classificationInterpretation object contains all the information needed to interperate the model.
    # These are the things I will be looking at in the next 4 lines of code.
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Creates a plot of the images which were misclassified most confidently.
    interp.plot_top_losses(9, figsize=(20,13))
    
    # Plots the confusion matrix.
    interp.plot_confusion_matrix(figsize=(20,20), dpi=120)
    
    # Same info as in confusion matrix but easier to see the worst cases.
    interp.most_confused(min_val=5)


