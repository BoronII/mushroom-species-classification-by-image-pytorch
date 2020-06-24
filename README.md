# mushroom-species-classification-by-image-pytorch
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mushroom  Species Classification v1 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The purpose of this project is for me to learn about deep learning with PyTorch, using the fastai front end. In this first version of the project I will restrict myself to the techniques introduced to me in in the first 3 lessons of the course [\"practical deep learning for coders\"](https://course.fast.ai/). In later versions of the project I intend to improve on the results that I achieve here and finally put the model into production. \n",
    "\n",
    "**Keywords:** neural nets, deep learning, pytorch, fastai, transfer learning, one cycle learning, boot strapping. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- introduce the problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Deleted:** cartoons, images of non mushrooms, far away images of mushrooms, images with logo/writing covering the photo, images with people featured prominently (images with hands/arms were left in), images of mushrooms split in half (only, I left it if it also contained a whole mushroom, pictures of just the stems (too close up), things that were splices of many images, images with objects obscuring the mushroom, black and white photos, duplicates, pictures of spores/spore prints, cooked mushrooms,  blurry, dried mushrooms, mushrooms in baskets/tupper ware, too close up, very low contrast with bg, not rotted, swiss army knives, cell phones, \n",
    "    - Did this realizing that it puts limitations on the kinds of images users can enter. Tried to do it so that I could keep the instructions for image entry simple and remain able to accept most reasonable images.\n",
    "    - I was doin this, deleting images that I thought it would be hard to learn from based on what they looked like to me (a human)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- choose mushroom species by looking here: "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Limitations"
   ]
  },
