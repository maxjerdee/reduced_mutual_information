# Reduced mutual information

##### Maximilian Jerdee, Alec Kirkley, Mark Newman

Implementation of the reduced mutual information, an information-theoretic similarity measure between two labelings of the same set of objects. This "reduced" version of the mutual information addresses the bias of the typical mutual information towards an excessive number of groups. This measure includes:
1. The refined Dirichlet-multinomial reduction described in Jerdee, Kirkley, and Newman (2024) 
2. The asymmetric normalization discussed in Jerdee, Kirkley, and Newman (2023) https://arxiv.org/abs/2307.01282

We provide implementations of these measures in python and c++, contained within the `\python` and `\cpp` directories. 

To 