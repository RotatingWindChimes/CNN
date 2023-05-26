# Run Experiment

To run this program and train the baseline model entirely, run ```python main.py```.

If you want to apply the data augmentation, such as __mixup__, run ```python main.py --type mixup``` in the command line. (__cutout__, __cutmix__ can be applied in the same way.)

If you want to train different model in a simple command, such as __baseline__, __mixup__, __cutmix__, __cutout__. run ```python main.py --type base cutmix cutout mixup``` or in any order you like.

If you have trouble reproducing any results, please raise a GitHub issue with your logs and results. Otherwise, if you have success, please share your trained model weights with us and with the broader community!

# Model Parameters

After you train the model, the parameters will be saved  as a file with the suffix pkl in the current directory.
