# Script demonstrating the calculation of the reduced mutual information
# between two labelings of the same set of objects.

# General imports
from os import listdir

# Local imports
from RMI import *

# Folder to read the example tests from
example_data_folder = "data"

# Iterate over the examples in the example_data_folder
for filename in listdir(example_data_folder):
    print("Reading",filename)
    # Read the two labelings from the given file
    # labels1, labels2 = read_labels(example_data_folder+"/"+filename)
    
    # # Calculate the reduced mutual information
    # rmi = reduced_mutual_information(labels1,labels2)
    # print("Reduced mutual information:",rmi)
    # print()

# Read the two labelings from the given fil


