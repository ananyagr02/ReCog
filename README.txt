Sign Language Detection in Python

This project implements a sign language detection system using Python, OpenCV, scikit-learn, TensorFlow, Mediapipe, and NumPy.
The project is developed using VS Code editor, project is divided into the following code sections:

	1.Data collection: This code collects images of sign language gestures using a webcam. The user can specify the class label of each gesture by pressing the key ‘Q’ on the keyboard.           	the code collects 100 images for each class label.
	2.Data preprocessing: This code preprocesses the collected images to make them suitable for training a machine learning model. The code performs the following operations: 
	 o Resizes the images to a fixed size
	 o Converts the images to grayscale
	 o Normalizes the pixel values
	3.Model training: This code trains a machine learning model to classify images of sign language gestures. The code uses TensorFlow, Scikit-Learn and Random Forest Classifier to    	train the model.
	4.Model inference: This code performs inference on the trained model to classify new images of sign language gestures. The code outputs the predicted class label for each input 	image.

To run the project, follow these steps:

1.Install the required Python packages: OpenCV, scikit-learn, TensorFlow, Mediapipe, and NumPy.
2.Clone this repository to your local machine.
3.Navigate to the project directory and run the following command in VSCode terminal to install the required project dependencies:
      pip install -r requirements.txt
4.To collect data, run the following command:
      python collect_imgs.py
5.To preprocess the collected data, run the following command:
      python create_dataset.py
6.To train the model, run the following command:
      python train_classifier.py
7.To make the model work, run the following command:
      python inference_classifier.py

The inference_classifier.py script will start a webcam and display the predicted class label for each frame. You can use this script to test the model on new images of sign language gestures.

Example:
To collect data for the letter "A", run the following command in VS Code terminal :

  python collect_imgs.py --class_label 0

This will start the webcam and collect 100 images of you portraying the sign language letter "A". Once you have collected data for all of the desired class labels, you can preprocess the data and train the model. To do this, run the following commands in VS Code terminal :

	python create_dataset.py
	python train_classifier.py

Once the model is trained, you can test it on new images of sign language gestures by running the following command:
	python inference_classifier.py

This will start the webcam and display the predicted class label for each frame.

Conclusion:

This project implements a simple but effective sign language detection system in Python. The system can be used to develop a variety of sign language-related applications, such as a sign language interpreter or a sign language learning tool.

