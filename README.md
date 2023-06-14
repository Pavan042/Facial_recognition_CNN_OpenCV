# Facial_recognition_CNN_OpenCV

### Introduction:
Emotion recognition using facial expressions is an important task in the field of computer vision and artificial intelligence. It involves analyzing and understanding the emotions conveyed by human faces through images. In this report, we will discuss the approach taken to build a model for emotion recognition using facial expressions.
### Data Preprocessing:
The dataset used for this task is the "fer2013" dataset, which contains facial expression images categorized into seven different emotions (Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral). The dataset was loaded into a pandas DataFrame using the 'read_csv' function. The shape of the DataFrame was checked to ensure the data was loaded correctly. The unique emotions present in the dataset were also inspected.
### Feature Extraction:
The raw pixel values of the images in the dataset were extracted as features. Each image in the dataset was represented as a 48x48 grayscale image with pixel values ranging from 0 to 255. To visualize the images, a subset of images from each emotion category was displayed using matplotlib.
To prepare the features for training, the pixel values were converted into a 3-channel RGB format. Each grayscale image was replicated across all three channels, resulting in a 3-channel image with identical pixel values in each channel. This conversion was done using the OpenCV library.
### Label Encoding:
The emotion labels in the dataset were encoded using the LabelEncoder from the scikit-learn library. The labels were transformed into numeric values ranging from 0 to 6 corresponding to the seven emotions. The encoded labels were further converted into one-hot encoded vectors using the np_utils.to_categorical function from Keras.
### Train-Test Split:
The dataset was split into training and validation sets using the train_test_split function from scikit-learn. The split was performed with a test size of 10% and with stratification to ensure an equal distribution of emotions in both sets. The shapes of the resulting training and validation sets were checked to verify the split.
### Model Architecture:
The VGG19 convolutional neural network (CNN) architecture was used as the base model for this task. The pre-trained weights from the ImageNet dataset were used to initialize the model. The last fully connected layer of the VGG19 model was removed, and a new global average pooling layer and a dense output layer with softmax activation were added to adapt the model for the emotion recognition task.
### Training:
The model was compiled with the categorical cross-entropy loss function and the Adam optimizer with a learning rate of 0.0001. Data augmentation techniques were applied using the ImageDataGenerator from Keras to introduce variations in the training images, such as rotation, shifting, shearing, and zooming. The generator was fitted on the training set to compute the necessary statistics for augmentation.
The model was trained using the fit function with the training data generator and the validation set. Early stopping and learning rate reduction callbacks were used to monitor the validation accuracy and adjust the learning rate, respectively. The training was performed for a total of 30 epochs with a batch size of 48.
### Model Evaluation:
The trained model's performance was evaluated using the validation set. The predictions were obtained using the model's predict function, and the predicted labels were compared to the true labels to calculate the classification metrics. The confusion matrix and classification report were generated using the scikit-plot library and the classification_report function from scikit-learn.
### Visualization and Analysis:
The model's performance was visualized using line plots to show the accuracy and loss values over epochs during training. The plots were saved as images for later reference. Additionally, a grid of randomly selected sad and happy images from the validation set was displayed along with their true and predicted labels to analyze the model's predictions qualitatively.
### Sample Predictions:
Lastly, some sample predictions were made using the trained model on a subset of validation images. The predicted labels were compared to the true labels, and the images along with the true and predicted labels were displayed. The process was repeated for a few samples, with a delay between each sample to observe the predictions interactively.
### Conclusion:
In this project, an approach to emotion recognition using facial expressions was implemented. The VGG19 CNN architecture was used as the base model, and the model was trained on the fer2013 dataset. The model achieved good accuracy on the validation set, as evident from the evaluation metrics and sample predictions. The preprocessing steps involved converting the grayscale images into 3-channel RGB format, label encoding, and train-test splitting. The model was trained using data augmentation techniques, and its performance was evaluated using standard classification metrics.
