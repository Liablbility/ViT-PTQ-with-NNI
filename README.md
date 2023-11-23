# ViT-PTQ-with-NNI
liablbility@github Nov 23 2023
a simple demo for ViT Post-Train-Qualification based on NNI framework
this demo aims to show the effectiveness of NNI for ViT PTQ task

we use the vit_b_16 so that this demo can be deployed on more resource-limited devices.
And we use the Imagenet pre-trained weights smaller dataset of stanford dog to make this demo a finetune task with less training time

Attention:
Put the data set of stanford dog in the directory of ./data or set the attribute 'download' of dataloader to be True

