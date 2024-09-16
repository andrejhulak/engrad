Enhanced micrograd version with convolution. I've implemented basic NN, LeNet (not entirely accurate nn part), AlexNet.
Working on the todo list below:

TO DO:
1. get a new dataset of colorful images - try 64x64, maybe 224x224 to fully test AlexNet
2. change update_weights - create something like .parameters() in PyTorch
3. performance - understand what's happening and what takes so long (AlexNet takes solid ~5s to process an "image")
4. implement VGG
5. ResNet later
6. figure out a way how to train them later on (as we will have a lot of images and the networks themselves are a little too large for my tiny laptop)
7. change how engine and the Tensor class behaves - maybe there's a way to speed things up there? (ties to 3. :) )
