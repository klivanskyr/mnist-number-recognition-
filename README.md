# mnist-number-recognition-
Uses PyTorch (CNN) and MNIST data set to identify numbers drawn. Let's user draw their own numbers.

$ pip install -r requirements.txt, to download all libraries used

Run $ python main.py -h, for help

Run $ python main.py --task train --num_epochs #, to train cnn. # represents an integer for the number of epochs.

Then to use the model,
Run $ python main.py --task draw --load_weights saved_weights/TIMESTAMP/FILENAME.pt
