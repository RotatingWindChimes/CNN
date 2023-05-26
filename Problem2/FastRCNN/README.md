# Run Experiments

To train the model entirely, run ```python main.py``` in the command line. The model will be trained in 20 epochs and for each epoch from the 7th, the model parameters will be saved in ```logs/FasterRCNN/saves/```, you can pick up the best model parameters after training.

Remarks: Before training, please place the VOC dataset under the file path ```data\voc```

# Test Model

If you want to test the model I have seleted (18 training epochs), you can detect unknown images in the following two steps:
* step 1: Please place the unknown image in the following file path ```data/unseen```, please keep sure that the category of objects you want to detect in the picture is one that already exists in the VOC dataset
* step 2: Enter ```python test.py --name filename``` in the command line. You can detect images in __any format__, such as jpg, jpeg, png, etc, just make sure the filename is complete and correct. The result will be saved as a __jpg image__ in the same path, and the image will start with the same name. (For example, if you have a picture named "03.jpg" in the ```unseen``` file, you can run ```python test.py --name 03.jpg``` to get the result. It will be saved as __03_pred.jpg__ in ```unseen``` file.)

Remarks: You can detect different images in the same time, just enter ```python test.py --name filename1 filename2...``` in the command line.

# Some Explanations
