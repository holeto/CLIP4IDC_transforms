from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch

def compute_metrics(x):
    # note: every sorted position whose value equals the diagonal is returned, so a row with tied
    # scores contributes several entries (e.g. identical texts in multi-sample TRANCE)
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T

def _rank_metrics(ranks):
    ranks = np.asarray(ranks)
    metrics = {}
    metrics['R1'] = float(np.sum(ranks < 1)) * 100 / len(ranks)
    metrics['R5'] = float(np.sum(ranks < 5)) * 100 / len(ranks)
    metrics['R10'] = float(np.sum(ranks < 10)) * 100 / len(ranks)
    metrics['MR'] = float(np.median(ranks)) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = float(np.mean(ranks)) + 1
    return metrics

def group_retrieval_metrics(sim_matrix, target_ids, group_ids):
    """Retrieval between transformations (texts) and groups of image pairs, for multi-sample TRANCE.

    sim_matrix: N x N, row i = text of sample i, column j = pair of sample j.
    target_ids: transformation of every sample; samples with the same id have the same text.
    group_ids: generation group (trans_id) of every sample; all members of a group have one target_id.

    A group is scored by the mean similarity over its members, which gives a K x G matrix (K distinct
    transformations, G groups; K == G unless a transformation was generated in two groups).
    Group to text: rank of the group's transformation among the K texts (one correct text).
    Text to group: rank of the best group of the transformation; its other groups are not counted as wrong.
    A rank is the number of wrong candidates with a strictly higher score, so ties add no entries.
    With one member per group and distinct texts this equals the per-sample metric (up to ties).
    """
    sim_matrix = np.asarray(sim_matrix, dtype=np.float64)
    target_ids, group_ids = np.asarray(target_ids), np.asarray(group_ids)
    assert sim_matrix.shape == (len(target_ids), len(target_ids)) and len(group_ids) == len(target_ids)

    keys, text_rows = np.unique(target_ids, return_index=True)
    groups, group_of_sample = np.unique(group_ids, return_inverse=True)
    group_key = np.empty(len(groups), dtype=target_ids.dtype)
    group_key[group_of_sample] = target_ids
    assert np.all(group_key[group_of_sample] == target_ids), "a group holds several target ids"

    membership = np.zeros((len(target_ids), len(groups)))
    membership[np.arange(len(target_ids)), group_of_sample] = 1.
    membership /= membership.sum(axis=0, keepdims=True)
    sim = sim_matrix[text_rows] @ membership                      # K x G
    correct = keys[:, np.newaxis] == group_key[np.newaxis, :]     # K x G

    correct_score = np.where(correct, sim, 0.).sum(axis=0)        # one correct text per group
    b2t_ranks = np.sum((sim > correct_score[np.newaxis, :]) & ~correct, axis=0)
    best_score = np.where(correct, sim, -np.inf).max(axis=1)
    t2b_ranks = np.sum((sim > best_score[:, np.newaxis]) & ~correct, axis=1)

    b2t_metrics, t2b_metrics = _rank_metrics(b2t_ranks), _rank_metrics(t2b_ranks)
    b2t_metrics["n"], t2b_metrics["n"] = len(groups), len(keys)
    return b2t_metrics, t2b_metrics
