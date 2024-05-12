# Reduced mutual information

##### Maximilian Jerdee, Alec Kirkley, Mark Newman

Implementation of the reduced mutual information, an information-theoretic similarity measure between two labelings of the same set of objects. This "reduced" version of the mutual information addresses the bias of the typical mutual information towards an excessive number of groups. Our recommended similarity measure includes:
1. The Dirichlet-multinomial reduction described in Jerdee, Kirkley, and Newman (2024) https://arxiv.org/pdf/2405.05393
2. The asymmetric normalization discussed in Jerdee, Kirkley, and Newman (2023) https://arxiv.org/abs/2307.01282

as well as alternate formulations of the mutual information.

Examples of labelings to be compared may be found in the `/data` directory. These consist of space separated `.txt` files where each line gives the true label of an object followed by the candidate label for that object. To test the python implementation on one of these data sets, we can for example run the script `run_RMI.py data/wines.txt`. 

More generally the `compute_RMI()` function of `RMI.py` may be used to compute the reduced mutual information under various reductions and normalizations given two lists of labels. 
