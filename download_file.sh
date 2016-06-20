#!/bin/bash

# 2016 Pattern Recognition Term Project
# Main Goals
# 1. Comparing two algorithms: ANN and SVM
# 2. Comparing the results driven by different data features.
#                                                                   Hyungwon Yang
#                                                                     SungSoo Kim
#                                                                      2016.06.05

# Windows
# 1. Please visit the github website and download it.
#    > https://github.com/hyung8758/ANN_SVM_comparison.git
# 2. Unzip the file and locate it to the directory in which the code
#    downloads the datasets and runs the training and testing process.
# 3. Find 'main_experiment.m' file and run step by step.


# Mac OSX and Linux
# Locate this shell script to the directory in which you want to
# proceed the codes and run it in the command line (terminal).
code_name="ANN_SVM_comparison"

if [ -d "$code_name" ]; then
    echo "Code package is already installed. Please run main_experiment.m on your matlab."
else
    git clone https://github.com/hyung8758/ANN_SVM_comparison.git
    echo "Code package was downloaded successfully. Please run main_experiment.m on your matlab."
fi
