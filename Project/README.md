## EE558 - A Network Tour of Data Science, Final Project Report

All the python source code is contained in the src folder.

The environment should contain the following libraries:
  - numpy
  - pandas
  - matplotlib
  - networkx
  - torch
  - geopy.distance
  
File description:
  - functions.py *contains all functions used in the main code.*
  - functions_nn.py *contains only the functions used for the neural net training.*
  - net.py *containes three neural net implementations of which only the GCN has proven useful.*
  - project.py *contains the main code used for this project. It is formatted to run in spyder and while you can run the entire file at once, it is highly recommended to only run the necessary parts from within an iPython console.*
  - trainNeuralNet.py *contains the code used to train the network. It relies also on functions_nn.py and net.py*
    
The trained neural nets used are located in 'trained neural nets/GCN'.

train_batch_nn_NTDS.sh is the bash file used to train the GCN on the SCITAS computing cluster.
For instructions on how to use the cluster visit https://www.epfl.ch/research/facilities/scitas/

The RE-Europe dataset used for this project is available at https://drive.google.com/drive/folders/1Ko7AVGlQ13z3_NtRpVLqDxhcv-TJRZjG
