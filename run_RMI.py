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

# Verbose run
compute_RMI(true_labels, candidate_labels, reduction='DM', normalization='asymmetric', verbose=True)
print()

# Compute RMIs
RMI_DM = compute_RMI(true_labels, candidate_labels)
RMI_flat = compute_RMI(true_labels, candidate_labels, reduction='flat')
RMI_unreduced = compute_RMI(true_labels, candidate_labels, reduction='none')

print(f"Filename: {filename}")
print(f"Asymmetrically normalized reduced mutual informations:")
print(f"RMI_DM: {RMI_DM:.3f}, RMI_flat: {RMI_flat:.3f}, RMI_unreduced: {RMI_unreduced:.3f}")

# Symmetric normalizations
RMI_DM_sym = compute_RMI(true_labels, candidate_labels, normalization='symmetric')
RMI_flat_sym = compute_RMI(true_labels, candidate_labels, reduction='flat', normalization='symmetric')
RMI_unreduced_sym = compute_RMI(true_labels, candidate_labels, reduction='none', normalization='symmetric')

print(f"Symmetrically normalized reduced mutual informations:")
print(f"RMI_DM: {RMI_DM_sym:.3f}, RMI_flat: {RMI_flat_sym:.3f}, RMI_unreduced: {RMI_unreduced_sym:.3f}")
