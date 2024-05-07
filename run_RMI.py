# Script to demonstrate the functionality of RMI.py

# General imports
import numpy as np
from sys import argv

# Local import
from RMI import compute_RMI, get_contingency_table, compute_RMI_from_contingency_table

# Load the labels
# Fix input to work with either Python 2 or 3
try:
    input = raw_input
except NameError:
    pass
# Get the filename
if len(argv)>1: filename = argv[1]
else: filename = input("Enter the name of the data file: ")

label_pairs = np.loadtxt(filename, dtype='str')
true_labels = label_pairs[:,0]
candidate_labels = label_pairs[:,1]

print(get_contingency_table(true_labels,candidate_labels))

print(compute_RMI(true_labels, candidate_labels, reduction='DM', normalization='asymmetric', verbose=True))