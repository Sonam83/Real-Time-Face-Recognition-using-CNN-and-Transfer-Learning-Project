# Real-Time-Face-Recognition-using-CNN-and-Transfer-Learning-Project
**About :** This project implements a real time face recognition system using Convolution Neural Networks(CNN) and Transfer Learning with pretrained models like **VGG16**, **ResNet50**, and **MobileNetV2** from keras applications

**Procedure :**
- Environment check using Anaconda prompt in Jupyter and Google colab
- Importing all the required Libraries
- Creating a dataset along with sub-folders classification as ["Angry","Sad","Sleep","Smile","Surprise"] using Webcam/Iriun Webcam/Video path
- Preprocessing involves Data Augmentation techniques like resizing, rescaling, normalisation adhered by splitting the data into training dataset,validation dataset and testing dataset using ImageDataGenerator
- Building model with transfer learning
- Hyperparameter tuning with Optuna
- Running optuna study to detect best parameters and best possible combinations
- saving the checkpoints and best model
- Evaluating the modelwith metrics confusion matrix, precision, Recall, F1 score, Accuracy
- Plotting train and validation loss , accuracy plots
- Live stream capturing video with opencv haarcascade to predict the class : Inference

**Tech-stack used :**
-Python 3.11.13
-Tensorflow 2.19.0
-Opencv
-Optuna
-Numpy,Matplotlib,Keras

**Summary :**
-The best model parameters out of all combinations recognized by Optuna are Best trial {'base_model': 'MobileNetV2', 'lr': 3.852277789415312e-05, 'freeze_all': True, 'freeze_except': 2, 'activation': 'relu', 'num_layers': 3, 'neurons': 64}

-The best value is 0.9977777600288391

-Training is stable , avoiding overfitting

-The model learns performs well on both training and validation data. -Final metrics: val_accuracy = 0.9978, accuracy = 0.9990

 **Acknowledgement :**
I would like to acknowledge the open source libraries Opencv,Tensorflow and Keras which played a significant role to streamline this face recognition project
