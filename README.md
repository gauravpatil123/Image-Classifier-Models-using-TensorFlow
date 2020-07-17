# Image-Classifier-Models-using-TensorFlow
Image classification models on various datasets using TenserFlow and Keras

**Models**
1. **Binary-Classifier:**
  - **Dataset**
    - [cats_and_dogs](https://www.kaggle.com/greg115/cats-and-dogs)
      1. Training Set Size = 20000 (cats = 10000, dogs = 10000)
      2. Validation Set Size = 5000 (cats = 2500, dogs = 2500)
      3. Test Set Size = 100 (mixed)
  - **Scripts**
    - BinaryImageGenerator : ImageDataGenerators for flowing the images to the model from dataset
    - DatasetDirectoryPreprocessing : Directories and processing of datasets
    - PlotCode : Plotting metrics from the trained model
  - **Classes**
    - BinaryCNN : model class for the convolutional neural network binary classifier 
  - **Executables**
    - TrainBinaryCNN : Training the BinaryCNN model using data generators from BinaryImageGenerator on the cats_and_dogs dataset
  - **Results**
    - Accuracy on Training and Validation set of the Binary Classifier
    - <img src="Binary-Classifier/Images/train_v_validation_accuracy.png" width=1000>
    - <img src="Binary-Classifier/Images/train_v_validation_loss.png" width=1000>
   
2. **Multi-Classifier:**
  - **Dataset**
    - [Monkey-Species](https://www.kaggle.com/slothkong/10-monkey-species?)
      1. Training Set Size = 1098 (spread across all 10 classes)
      2. Validation Set Size = 272 (spread across all 10 classes)
    - [Testing Dataset](https://github.com/gauravpatil123/Image-Classifier-Models-using-TensorFlow/tree/working/Multi-Classifier/data/testing)
      1. Testing Set Size = 30 (3 images of each class)
  - **Scripts**
    - MultiCNN : model for the convolutional neural network multi classifier
    - MultiImageGenerator : ImageDataGenerators for flowing the images to the model from dataset
    - DatasetDirectoryPreprocessing : Configuring the directories and processing the dataset
    - PlotCode : Plotting metrics from the trained model
  - **Executable**
    - TrainMultiCNN : Training the MultiCNN model using data generators from MultiImageGenerator on the Monkey-Species dataset
    - TestMultiCNN : Testing the trained MultiCNN model on the Testing Dataset
  - **Results**
    - Accuracy on Training and Validation set of Multi Classifier
    - <img src="Multi-Classifier/Images/train_v_validation_accuracy.png" width=1000>
    - <img src="Multi-Classifier/Images/train_v_validation_loss.png" width=1000>
  
3. **Multi-Classifier-with-Transfer-Learning:**
  - **Dataset**
    - [Monkey-Species](https://www.kaggle.com/slothkong/10-monkey-species?)
      1. Training Set Size = 1098 (spread across all 10 classes)
      2. Validation Set Size = 272 (spread across all 10 classes)
    - [Testing Dataset](https://github.com/gauravpatil123/Image-Classifier-Models-using-TensorFlow/tree/working/Multi-Classifier-with-Transfer-Learning/data/testing)
      1. Testing Set Size = 30 (3 images of each class)

  - **Scripts**
    - PreTrainedInceptionV3 : importing and loading the pre-trained weights from the InceptionV3 model from keras.applications and configuring the model
    - MultiCNN : configuring a model by adding Dense and output layers at the bootom of the pre-trained model
    - ImageGenerators : ImageDataGenerators for flowing theimages to the model from dataset
    - DatasetDirectoryPreprocessing : Directories, processing and splitting of dataset in to training, validation and test sets
    - PlotCode : Plotting metrics from the trained model
  - **Executables**
    - TrainMultiCNN : Training the neural network using data generators from ImageGenerators on the Monkey-Species dataset
    - TestMultiCNN : Testing the MultiCNN model on the Testing Dataset
  - **Results**
    - Accuracy on Training and Validation set of the neural network
    - <img src="Multi-Classifier-using-Transfer-Learning/Images/train_v_validation_accuracy.png" width=1000>
    - <img src="Multi-Classifier-using-Transfer-Learning/Images/train_v_validation_loss.png" width=1000>
