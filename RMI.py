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

    # Compute summary information
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

def _log_binom(a,b):
    """log(binomial(a,b)), overflow protected
    """
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a-b+1)

def _log_Omega_EC(rs,cs,useShortDimension=True,symmetrize=False):
    """Approximate the log of the number of contingency tables with given row and column sums with the EC estimate of Jerdee, Kirkley, Newman (2022) https://arxiv.org/abs/2209.14869

    :param rs: row sums
    :type rs: list, int
    :param cs: column sums
    :type cs: list, int
    :param useShortDimension: Whether to optimize the encoding by possibly swapping the definitions of rows and columns, defaults to True
    :type useShortDimension: bool, optional
    :param symmetrize: Whether to symmetrize the estimate, defaults to False
    :type symmetrize: bool, optional
    :return: estimate of log_Omega
    :rtype: float
    """
    rs = np.array(rs)
    cs = np.array(cs)
    # Remove any zeros
    rs = rs[rs > 0]
    cs = cs[cs > 0]
    if len(rs) == 0 or len(cs) == 0:
        return -np.inf # There are no tables
    if useShortDimension: # Perfomance of the EC estimate is generally improved when there are
                            # more rows than columns. If this is not the case, swap definitions around
        if len(rs) >= len(cs):
            return _log_Omega_EC(rs,cs,useShortDimension=False)
        else:
            return _log_Omega_EC(cs,rs,useShortDimension=False)
    else:
        if symmetrize:
            return (_log_Omega_EC(rs,cs,symmetrize=False)+log_Omega_EC(cs,rs,symmetrize=False))/2
        else:
            m = len(rs)
            N = sum(rs)
            if N == len(cs): # In this case, we may simply return the exact result (equivalent to alpha = inf)
                return gammaln(N + 1) - sum(gammaln(rs + 1))
            alphaC = (N**2-N+(N**2-sum(cs**2))/m)/(sum(cs**2)-N)
            result = -_log_binom(N + m*alphaC - 1, m*alphaC - 1)
            for r in rs:
                result += _log_binom(r + alphaC - 1,alphaC-1)
            for c in cs:
                result += _log_binom(c + m - 1, m - 1)
            return result

def _compute_flat_subleading_terms(contingency_table):
    """Compute the subleading contributions to the entropies in the flat reduction.

    :param contingency_table: Contingency table of label cooccurrences between the true and candidate labels.
    :type contingency_table: np.array
    :return: Change in H(g), change in H(c), change in H(g|c), change in H(c|g), change in H(g|g), change in H(c|c)
    :rtype: float, float, float, float, float, float
    """
    # Compute summary information
    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    qg = len(ng)
    qc = len(nc)

    # H(ng) = H(qg) + H(ng|qg)
    delta_Hg = np.log(n) + _log_binom(n - 1, qg - 1)
    delta_Hc = np.log(n) + _log_binom(n - 1, qc - 1)
    delta_HgGc = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, nc)
    delta_HcGg = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, ng)
    delta_HgGg = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, ng)
    delta_HcGc = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, nc)
    
    # Convert to base 2
    delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = delta_Hg/np.log(2), delta_Hc/np.log(2), delta_HgGc/np.log(2), delta_HcGg/np.log(2), delta_HgGg/np.log(2), delta_HcGc/np.log(2)

    return delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc

def _compute_DM_subleading_terms(contingency_table):
    """Compute the subleading contributions to the entropies in the Dirichlet-multinomial reduction. 

    :param contingency_table: Contingency table of label cooccurrences between the true and candidate labels.
    :type contingency_table: np.array
    :return: Change in H(g), change in H(c), change in H(g|c), change in H(c|g), change in H(g|g), change in H(c|c)
    :rtype: float, float, float, float, float, float
    """
    # Compute summary information
    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    qg = len(ng)
    qc = len(nc)

    # H(ng) = H(qg) + H(ng|qg)
    delta_Hg = np.log(n) + _log_binom(n - 1, qg - 1)
    delta_Hc = np.log(n) + _log_binom(n - 1, qc - 1)
    delta_HgGc = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, nc)
    delta_HcGg = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, ng)
    delta_HgGg = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, ng)
    delta_HcGc = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, nc)

    # Convert to base 2
    delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = delta_Hg/np.log(2), delta_Hc/np.log(2), delta_HgGc/np.log(2), delta_HcGg/np.log(2), delta_HgGg/np.log(2), delta_HcGc/np.log(2)

    return delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc

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
        print()

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
    HgGg = 0 # H(g|g)
    HcGc = 0 # H(c|c)

    if verbose:
        print(f"Unreduced mutual information (bits): I(g;c) = {I:.3f}")
        print(f"Unreduced entropies (bits): H(g) = {Hg:.3f}, H(c) = {Hc:.3f}")
        print(f"Unreduced conditional entropies (bits): H(g|c) = {HgGc:.3f}, H(c|g) = {HcGg:.3f}")
        print()
    
    if reduction == 'none':
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = 0, 0, 0, 0, 0, 0
    if reduction == 'flat':
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = _compute_flat_subleading_terms(contingency_table)
    if reduction == 'DM':
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = _compute_DM_subleading_terms(contingency_table)
    
    if verbose:
        print(f"Subleading terms in the entropies:")
        print(f"delta_H(g) = {delta_Hg:.3f}, delta_H(c) = {delta_Hc:.3f}, delta_H(g|c) = {delta_HgGc:.3f}, delta_H(c|g) = {delta_HcGg:.3f}, delta_H(g|g) = {delta_HgGg:.3f}, delta_H(c|c) = {delta_HcGc:.3f}")
        print()

    # Adjust the leading behavior with the subleading terms
    Hg += delta_Hg
    Hc += delta_Hc
    HgGc += delta_HgGc
    HcGg += delta_HcGg
    HgGg += delta_HgGg
    HcGc += delta_HcGc

    # Compute the reduced mutual information(s)
    RMI_g_c = Hg - HgGc
    RMI_c_g = Hc - HcGg
    RMI_g_g = Hg - HgGg
    RMI_c_c = Hc - HcGc

    if verbose:
        print(f"Reduced mutual information (bits): I(g;c) = {RMI_g_c:.3f}")
        print(f"Full entropies (bits): H(g) = {Hg:.3f}, H(c) = {Hc:.3f}")
        print(f"Full conditional entropies (bits): H(g|c) = {HgGc:.3f}, H(c|g) = {HcGg:.3f}")
        print()

    # (Potentially) normalize the mutual information
    if normalization == 'none':
        RMI = RMI_g_c # Note that we do not symmetrize the mutual information in this case
    if normalization == 'asymmetric':
        RMI = RMI_g_c/RMI_g_g
    if normalization == 'symmetric':
        RMI = (RMI_g_c + RMI_c_g)/(RMI_g_g + RMI_c_c)

    if verbose:
        print(f"Normalized reduced mutual information NMI(g;c) = {RMI:.3f}")

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