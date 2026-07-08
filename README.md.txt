# Deep Learning – Assignment 1  
*Course:* AI2100   
*Name:* G.Geetha charani 
*Roll Number:* AI24BTECH11013

Repository Structure

A1_AI24BTECH11013
│
├── Qn_1
│   └── ANN_backpropagation.ipynb
│
├── Qn_2
│   └── 2_qn.pdf
│
├── Qn_3
│   └── 3_qn.pdf
│
├── Qn_4
│   ├── dataset_description.pdf
│   ├── datageneration.py
│   ├── mess_dataset_500.csv
│   └── mess_dataset_5000.csv
│
├── Qn_5
│   ├── 5_qn.pdf
│   ├── mess_dataset_5000.csv
│   ├── Adaline_experiments.ipynb
│   └── _Adaline.py
│
├── Qn_6
│   ├── 6_experiments.ipynb
│   ├── 6_qn
│   ├── mess_dataset_5000.csv
│   ├── mlp.py
│   ├── losses.py
│   ├── weights.py
│   ├── activations.py
│   ├── optimizers.py
│   └── _Adaline.py
│
└── Qn_7
    ├── mess_dataset_500.csv
    ├── 7_kernals
    └── 7_qn

Installation

Python 3.9+ is recommended.

Install the required packages:
bash pip install numpy pandas matplotlib jupyter
Question-wise Instructions
Q1 – Artificial Neural Network

File

Qn_1/ANN_backpropagation.ipynb

Run

cd Qn_1
jupyter notebook ANN_backpropagation.ipynb

This notebook implements an Artificial Neural Network trained using backpropagation.

Q2 – Universal Approximation Theorem

File

Qn_2/2_qn.pdf

Contains theoretical answers related to the Universal Approximation Theorem.

Q3 – One Step of Gradient Descent

File

Qn_3/3_qn.pdf

Contains manual derivations for:

forward pass

loss computation

gradient calculations

parameter updates using gradient descent

Q4 – Dataset Generation

Files

Qn_4/
datageneration.py
dataset_description.pdf
mess_dataset_500.csv
mess_dataset_5000.csv

Run

cd Qn_4
python datageneration.py

This script generates the datasets used in later questions:

mess_dataset_500.csv
mess_dataset_5000.csv
Q5 – ADALINE

Files

Qn_5/
_Adaline.py
Adaline_experiments.ipynb
5_qn.pdf
mess_dataset_5000.csv

Run

cd Qn_5
jupyter notebook Adaline_experiments.ipynb

Uses dataset:

mess_dataset_5000.csv
Q6 – Multi Layer Perceptron

Files

Qn_6/
mlp.py
activations.py
losses.py
optimizers.py
weights.py
6_experiments.ipynb
6_qn
mess_dataset_5000.csv
_Adaline.py

Run

cd Qn_6
jupyter notebook 6_experiments.ipynb

Uses dataset:

mess_dataset_5000.csv

Q7 – Neural Network Features for Kernel Methods

Files

Qn_7/
7_kernals
7_qn
mess_dataset_500.csv

This section uses neural network representations as features for kernel methods.

Uses dataset:

mess_dataset_500.csv

Note:
All neural network algorithms are implemented from scratch using NumPy.

No deep learning libraries such as PyTorch or TensorFlow are used.

Random seeds are fixed where necessary to ensure reproducibility.

All visualizations are generated within Jupyter notebooks.

Ensure to add the contents in the notebook while running.