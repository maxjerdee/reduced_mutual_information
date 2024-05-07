# Script to compute the reduced mutual information

import numpy as np
import pandas as pd
from scipy.special import gammaln

def _compute_I_Hg_Hc(contingency_table):
    """Compute the unreduced microcanonical mutual information and entropies of the ground truth and candidate labelings.

    :param contingency_table: Contingency table of label cooccurrences between the true and candidate labels
    :type contingency_table: np.array
    :return: Mutual information, entropy of the ground truth labels, entropy of the candidate labels
    :rtype: float, float, float
    """

    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    qg = len(ng)
    qc = len(nc)

    I = gammaln(n + 1) - np.sum(gammaln(ng + 1)) - np.sum(gammaln(nc + 1)) + np.sum(gammaln(contingency_table.flatten() + 1))
    Hg = gammaln(n + 1) - np.sum(gammaln(ng + 1))
    Hc = gammaln(n + 1) - np.sum(gammaln(nc + 1))

    # Convert to bits (log base 2)
    I /= np.log(2)
    Hg /= np.log(2)
    Hc /= np.log(2)

    return I, Hg, Hc



def _compute_flat_subleading_terms(contingency_table):
    """Compute the subleading contributions to the entropies in the flat reduction.

    :param contingency_table: Contingency table of label cooccurrences between the true and candidate labels.
    :type contingency_table: np.array
    :return: Change in H(g), change in H(c), change in H(g|c), change in H(c|g)
    :rtype: float, float, float, float
    """
    
    return delta_Hg, delta_Hc, delta_HgGc, delta_HcGg

def get_contingency_table(true_labels, candidate_labels):
    """Generate the contingency table for the true and candidate labels (uses pd.crosstab)

    :param true_labels: True labels of the objects
    :type true_labels: list
    :param candidate_labels: Candidate labels of the objects
    :type candidate_labels: list
    :return: DataFrame with the contingency table
    :rtype: pd.DataFrame
    """

    assert len(true_labels) == len(candidate_labels), "The number of true and candidate labels must be the same."
    
    return pd.crosstab(true_labels, candidate_labels, rownames=['true_labels'], colnames=['candidate_labels'])

def compute_RMI_from_contingency_table(contingency_table, reduction='DM', normalization='asymmetric', verbose=False):
    """Compute the reduced mutual information from the contingency table. True labels denote rows of this table. Returned in bits (log base 2) if unnormalized.
    :param contingency_table: Contingency table of label cooccurrences between the true and candidate labels
    :type contingency_table: np.array | pd.DataFrame
    :param reduction: Type of reduction of the mutual information, defaults to 'DM'. Options: {'DM', 'flat', 'none'}
    :type reduction: str, optional
    :param normalization: Type of normalization of the mutual information, defaults to 'asymmetric'. Options: {'asymmetric', 'symmetric', 'none'}
    :type normalization: str, optional
    :param verbose: Whether to print extra information in the computation process, defaults to False.
    :type verbose: bool, optional
    :return: Reduced mutual information
    :rtype: float
    """
    assert reduction in ['DM', 'flat', 'none'], "The reduction must be one of {'DM', 'flat', 'none'}"
    assert normalization in ['asymmetric', 'symmetric', 'none'], "The normalization must be one of {'asymmetric', 'symmetric', 'none'}"

    if verbose:
        print("Contingency table:")
        print(contingency_table)

    if type(contingency_table) == pd.DataFrame:
        contingency_table = contingency_table.to_numpy()
    
    assert type(contingency_table) == np.ndarray, "The contingency table must be a numpy array or a pandas DataFrame."

    # Drop any empty rows or columns from the contingency table
    contingency_table = contingency_table[~np.all(contingency_table == 0, axis=1)]
    contingency_table = contingency_table[:, ~np.all(contingency_table == 0, axis=0)]
    
    I, Hg, Hc = _compute_I_Hg_Hc(contingency_table)
    # Computing the other unreduced information costs from this information
    HgGc = Hg - I # H(g|c)
    HcGg = Hc - I # H(c|g)

    if verbose:
        print(f"Unreduced mutual information (bits): I = {I:.3f}, Entropy of the true labels H(g) = {Hg:.3f}, Entropy of the claimed labels H(c) = {Hc:.3f}")
        print(f"Unreduced conditional entropies (bits): H(g|c) = {HgGc:.3f}, H(c|g) = {HcGg:.3f}")
    
    
    RMI = 0
    return RMI

def compute_RMI(true_labels, candidate_labels, reduction='DM', normalization='asymmetric', verbose=False):
    """Compute the reduced mutual information between the true and candidate labels. Returned in bits (log base 2) if unnormalized.

    :param true_labels: True labels of the objects
    :type true_labels: list
    :param candidate_labels: Candidate labels of the objects
    :type candidate_labels: list
    :param reduction: Type of reduction of the mutual information, defaults to 'DM'. Options: {'DM', 'flat', 'none'}
    :type reduction: str, optional
    :param normalization: Type of normalization of the mutual information, defaults to 'asymmetric'. Options: {'asymmetric', 'symmetric', 'none'}
    :type normalization: str, optional
    :param verbose: Whether to print extra information in the computation process, defaults to False.
    :type verbose: bool, optional
    :return: Reduced mutual information
    :rtype: float
    """
    assert len(true_labels) == len(candidate_labels), "The number of true and candidate labels must be the same."
    assert reduction in ['DM', 'flat', 'none'], "The reduction must be one of {'DM', 'flat', 'none'}"
    assert normalization in ['asymmetric', 'symmetric', 'none'], "The normalization must be one of {'asymmetric', 'symmetric', 'none'}"

    contingency_table = get_contingency_table(true_labels, candidate_labels)
    RMI = compute_RMI_from_contingency_table(contingency_table, reduction=reduction, normalization=normalization, verbose=verbose)
    return RMI