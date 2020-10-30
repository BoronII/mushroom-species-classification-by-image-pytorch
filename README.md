# mushroom-species-classification-by-image-pytorch
Full details of the project is presented in the exposition notebook.
The purpose of this project is for me to learn about deep learning with 
PyTorch, using the fastai front end. 

Keywords: neural nets, deep learning, pytorch, fastai, transfer learning, 
fine tuning, one cycle learning, boot strapping, fine grain classification. 

## Introduction 

Being able to tell one mushroom species from another can mean the difference 
between picking yourself a tasty meal or ending up with something that could 
make you sick or even lead to death. Approximately 14,000 species of mushrooms 
have been cataloged worldwide and creating a classifier that has been trained 
on all of these species presents a data collection challenge that I was not 
prepared to undertake. In this project, I wanted to perform the data collection 
and data cleaning steps myself, to keep things manageable I decided to work 
toward creating a classifier that will be useful to people foraging in a more 
localized geographical area. I learned that mushroom foraging is a popular 
activity in the Canadian province of British Columbia. Being a Canadian myself, 
this is the geographical region I decided to focus my efforts on.

## Data acquisition and cleaning

The top 300 images (using the species name as the search term) were downloaded from bing image search using [Google/Bing Images Web Downloader](https://github.com/ultralytics/google-images-download).

This tool had the advantage of allowing me to automate the gathering of a data set containing over 22,000 images in 76 categories. However, this automation comes at the cost of introducing a substantial amount of noise into the data set. The top 300 image search results included the following types of undesired images:

    1. Cartoon images
    2. Images that did not contain mushrooms
    3. Images that contained more than one type of mushroom
    4. Images that were too zoomed out
    5. Images that were too close up
    6. Images with writing or objects obscuring the mushrooms
    7. Images that featured objects such as people, swiss army knives, baskets, tupperware, cell phones
    8. Images that did not contain any intact mushrooms
    9. Black and white images
    10. duplicate images
    11. Images of cooked mushrooms
    12. Images of moldy mushrooms
    13. Blurry images 
    14. Images that were too small
    15. Low contrast images

My reasoning for removing these images was that they would be misleading the classifier in some way.

The initial data set was cleaned by me, I made two passes through the entire set of images, looking at each image, and deleting any misleading images. In the process of doing this, three of the categories ended up with fewer than 50 images, these categories were excluded from the final data set. The final data set contains 16729 images from 73 categories, 80 percent of the images were used for training, the remaining 20 percent was used for validation.

## Modeling results

The plot belows shows the images from the validation set for which the loss is greatest, along with their predicted class, actual class, loss, and probability of actual class.

![top_losses.png](https://github.com/BoronII/mushroom-species-classification-by-image-pytorch/blob/master/images/top_losses.png)







