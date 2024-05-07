from __future__ import print_function,division

# Program for calculating the reduced mutual information of Newman,
# Cantwell, and Young (2019) for two clusterings read from a file
# updated with the improved approximation of Jerdee, Kirkley, and Newman (2023)
#   for the number of contingency tables
#
# This program takes one argument: the name of the file
#
# Data format for the file is two columns of group labels, separated
# by whitespace (spaces, tabs). Note that for asymmetric normalizations, the second labelling
# is taken to be the ground truth labelling.
# For instance:
#
# 0 0
# 0 0
# 0 1
# 1 2
# 1 0
# 1 3
# etc.
#
# Group labels do not need to be integers (although should not contain whitespace)
#
# Written by Max Jerdee  24 JUL 2023 (Adapted from Mark Newman's code)

from sys import argv
from math import log,exp
from numpy import zeros,sum,array
from scipy.special import gammaln

# Whether to print extra information about the computation process
verbose = True

# Fix input to work with either Python 2 or 3
try:
    input = raw_input
except NameError:
    pass

# Get the filename
if len(argv)>1: filename = argv[1]
else: filename = input("Enter the name of the data file: ")

# Read the data from the file
f_in = open(filename, "r")
# Running counts of the number of groups in each labelling 
q1 = 0
q2 = 0
# Dictionaries which associate an index to each label (for each labelling)
labels_dict_1 = {} 
labels_dict_2 = {} 
data = [] # array of the labels, converted into label indices
for line in f_in.readlines():
    line_split = line.split()
    assert len(line_split) == 2, f"Issue reading line: {line}"
    if line_split[0] not in labels_dict_1.keys(): # If an observed label is new, assign it an index
        labels_dict_1[line_split[0]] = q1
        q1 += 1
    if line_split[1] not in labels_dict_2.keys(): # If an observed label is new, assign it an index
        labels_dict_2[line_split[1]] = q2
        q2 += 1
    data.append((labels_dict_1[line_split[0]],labels_dict_2[line_split[1]]))
n = len(data)

if verbose:
    print("Read",n,"objects with q1 =",q1,"and q2 =",q2)
    print()

# Construct the contingency table
n12 = zeros([q1,q2],int)
for k in range(n):
    r,s = data[k]
    n12[r,s] += 1
n1 = sum(n12,axis=1)
n2 = sum(n12,axis=0)

if verbose:
    print("Contingency table:")
    print(n12)
    print()
    print("Row sums:   ",n1)
    print("Column sums:",n2)
    print()

# Calculate the standard mutual information (and indiviudal informations) in bits
I = gammaln(n+1)
H1 = gammaln(n+1)
H2 = gammaln(n+1)
for r in range(q1):
    for s in range(q2):
        I += gammaln(n12[r,s]+1)
for r in range(q1): 
    I -= gammaln(n1[r]+1) 
    H1 -= gammaln(n1[r]+1)
for s in range(q2): 
    I -= gammaln(n2[s]+1) 
    H2 -= gammaln(n2[s]+1)

if verbose:
    print("Mutual information I =",I,"total bits (natural base)")
    print("Entropy H1 =",H1,"total bits")
    print("Entropy H2 =",H2,"total bits")
    print("Symmetric NMI, I/((H1+H2)/2) =",2*I/(H1+H2))
    print("Asymmetric NMI (labelling 2 taken as truth), I/H2 =",I/H2)
    print()

# log(binom(a,b)), but avoiding overflow
def logBinom(a,b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a-b+1)

def logOmegaEC(rs,cs,useShortDimension=True,symmetrize=False):
    rs = array(rs)
    cs = array(cs)
    if useShortDimension: # Perfomance of the EC estimate is generally improved when there are
                            # more rows than columns. If this is not the case, swap definitions around
        if len(rs) >= len(cs):
            return logOmegaEC(rs,cs,useShortDimension=False)
        else:
            return logOmegaEC(cs,rs,useShortDimension=False)
    else:
        if symmetrize:
            return (logOmegaEC(rs,cs,symmetrize=False)+logOmegaEC(cs,rs,symmetrize=False))/2
        else:
            m = len(rs)
            N = sum(rs)
            if N == len(cs): # In this case, we may simply return the exact result (equivalent to alpha = inf)
                return gammaln(N + 1) - sum(gammaln(rs + 1))
            alphaC = (N**2-N+(N**2-sum(cs**2))/m)/(sum(cs**2)-N)
            result = -logBinom(N + m*alphaC - 1, m*alphaC - 1)
            for r in rs:
                result += logBinom(r + alphaC - 1,alphaC-1)
            for c in cs:
                result += logBinom(c + m - 1, m - 1)
            return result

# Calculate the corrections
logOmega12 = logOmegaEC(n1,n2)
logOmega11 = logOmegaEC(n1,n1)
logOmega22 = logOmegaEC(n2,n2)
RMI = I - logOmega12
RH1 = H1 - logOmega11
RH2 = H2 - logOmega22

if verbose:
    print("Estimated (log) number of contingency tables log(Omega) =",logOmega12,"bits")
    print("RMI =",RMI,"total bits")
    print("Symmetrically normalized RMI =",2*RMI/(RH1+RH2))
print("Asymmetrically normalized RMI =",RMI/RH2)