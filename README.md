# cnn_building_-recogniser

cnn for building recognition using pytorch. The building image dataset Zubud has been used to test this implementation. 
This implementation was developed by Lawrence Chen and Isaac Chen in 2019, with the inspiration of ANU CSIT professor Tom Gedeon's labs. 

## Files of the project
1. cnn.py - the actual cnn implementation in pytorch
2. processing.py - preprocessing the data (may need to comment/uncomment code to achieve the result you want)
3. checkpoint12.pth - A checkpoint for the trained cnn (may need to comment/uncomment code if you want to retrain the cnn entirely)

## Steps for CNN recognition:
0. To run this program, opencv and pytorch package should be installed in advance. 
1. Download the Zubud dataset and Zubud query image from http://www.vision.ee.ethz.ch/showroom/zubud/
2. After unzipping the dataset, run processing.py to divide database images into training set and testing set by the proportion of 80:20 and change the size to be 480x480. The folder of training set is “png-Zubud/train”, and the folder of testing set is “png-Zubud/test”.
3. Run cnn.py to see the performance on recognising database images (make sure checkpoint12 is under the same folder as cnn.py).
4. To see the performance of CNN on query images, replace the folder “png-Zubud/test” with the  folder “qimage” first and then rename the folder “qimage” as “test”. run cnn.py to see the performance on recognising query images.

Note: the folder “qimage” contains the labelled Zubud query images and can be used to test the real performance of the CNN, as the original Zubud query image is unlabelled. 

With the provided checkpoint, this implementation hits 74% accuracy for recognising the building in Zubud dataset. 
![Screenshot](https://i.imgur.com/0kkB20g.png)