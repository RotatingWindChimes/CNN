# Run Experiment

This program trains and saves the ResNet18 model on the CIFAR-100 data set under different methods and different training times. To specify how the data is enhanced and the number of iterations, use the command line arguments __type__ and __time__.

For example, if you want to apply the data augmentation , such as __mixup__, and the training epochs is set to be 200, run ```python main.py --type mixup --time 200``` in the command line. (__cutout__, __cutmix__ can be applied in the same way.)

If you want to train different model in a simple command, such as __baseline__, __mixup__, __cutmix__, __cutout__. run ```python main.py --type base cutmix cutout mixup --time 200 100 200 100``` or in any order you like.

If you have trouble reproducing any results, please raise a GitHub issue with your logs and results. Otherwise, if you have success, please share your trained model weights with us and with the broader community!

# Model Parameters

After you train the model, the parameters will be saved  as a file with the suffix pkl in the current directory.

# Test Model

