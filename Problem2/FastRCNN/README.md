# Run Experiments

To train the model entirely, run ```python main.py``` in the command line. The model will be trained in 20 epochs and for each epoch from the 7th, the model parameters will be saved in ```logs/FasterRCNN/saves/```, you can pick up the best model parameters after training.

Remarks: Before training, please place the VOC dataset under the file path ```data/voc```, the dataset is available at the Baidu Web disk link below.

# Test Model

If you want to test the model I have seleted (18 training epochs), you can detect unknown images in the following two steps:
* step 1: Please place the unknown image in the following file path ```data/unseen```, please keep sure that the category of objects you want to detect in the picture is one that already exists in the VOC dataset
* step 2: Enter ```python test.py --name filename``` in the command line. You can detect images in __any format__, such as jpg, jpeg, png, etc, just make sure the filename is complete and correct. The result will be saved as a __jpg image__ in the same path, and the image will start with the same name. (For example, if you have a picture named "03.jpg" in the ```unseen``` file, you can run ```python test.py --name 03.jpg``` to get the result. It will be saved as __03_pred.jpg__ in ```unseen``` file.)

Remarks: 
* Before training, please place the model parameters under the file path ```logs/FasterRCNN/saves/```, the parametes are available at the Baidu Web disk link below.
* You can detect different images in the same time, just enter ```python test.py --name filename1 filename2...``` in the command line.

# Some Explanations
* File ```data```
* * ```save```: It contains the origin pictures and the proposal boxes required in the first part.
* * ```unseen```: It contains the unknown pictures for detection.
* * ```voc```: It contains the VOC dataset to be trained.

* File ```logs```
* * It contains the model parameters after each training epoch in ```logs/FasterRCNN/saves```
* * The results for tensorboard visualization are also stored in this file
