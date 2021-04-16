# Musical Mood Classifier
Code for the class project in EECS 351 (Digital Signal Processing). Group members include Jack Smith, Jorge Alvarez, and Saaketh Medepalli.

# Dependencies  
This project has only been tested with the following:  
* NumPy Version 1.19.5   
* Torch Version 1.8.1
* Python >3.8.5

# Instructions for Testing
We used Visual Studio Code for testing. Open featurizer_tests.py and run entire file. This provides plots for the three features we have coded so far, and for one song of each mood.

To test our different classifier methods:
1. Open the terminal and type: python3 classifier_tests.py
2. It will prompt user input for a path to a song, we provide 4 sample songs, one for each mood. Type one of: test_songs/test_happy.wav, test_songs/test_sad.wav, test_songs/test_calm.wav, test_songs/test_hype.wav
3. It will then output the prediction from each of our 3 methods: neural network, k nearest neighbors, and SVM.
