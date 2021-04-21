# Musical Mood Classifier
Code for the class project in EECS 351 (Digital Signal Processing). Group members include Jack Smith, Jorge Alvarez, and Saaketh Medepalli.

# Dependencies  
This project has only been tested with the following:  
* NumPy Version 1.19.5   
* Torch Version 1.8.1
* Python >3.8.5

# Dataset
We gathered the dataset ourselves (total dataset not here because it is too large) by using a Python library called Savify. This allowed us to download playlists categorized by each mood from Spotify. The playlists we used are given below:

**Happy:** https://open.spotify.com/playlist/4AnAUkQNrLKlJCInZGSXRO  
**Sad:** https://open.spotify.com/playlist/78FHjijA1gBLuVx4qmcHq6  
**Calm:** https://open.spotify.com/playlist/6EIVswdPfoE9Wac7tB6FNg and https://open.spotify.com/playlist/37i9dQZF1DX5bjCEbRU4SJ  
**Hype/Energetic:** https://open.spotify.com/playlist/37i9dQZF1DX4eRPd9frC1m  

# Instructions for Testing
Open featurizer_tests.py and run entire file. This provides plots for the three features we have coded so far, and for one song of each mood.

To test our different classifier methods:
1. Open the terminal and type: python3 classifier_tests.py
2. It will prompt user input for a path to a song, we provide 4 sample songs, one for each mood. Type one of: test_songs/test_happy.wav, test_songs/test_sad.wav, test_songs/test_calm.wav, test_songs/test_hype.wav
3. It will then output the prediction from each of our 3 methods: neural network, k nearest neighbors, and SVM.

# Website
https://saakethm.wixsite.com/mood-classifier
