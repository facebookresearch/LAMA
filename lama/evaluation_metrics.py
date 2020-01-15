# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
import scipy


def __max_probs_values_indices(masked_indices, log_probs, topk=1000):

    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    return log_probs, index_max_probs, value_max_probs


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg


def get_ranking(log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=True):

    experiment_result = {}

    log_probs, index_max_probs, value_max_probs = __max_probs_values_indices(masked_indices, log_probs, topk=topk)
    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    if print_generation:
        print(return_msg)

    MRR = 0.
    P_AT_X = 0.
    P_AT_1 = 0.
    PERPLEXITY = None

    if label_index is not None:

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)

        query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
        ranking_position = (index_max_probs==query).nonzero()

        # LABEL PERPLEXITY
        tokens = torch.from_numpy(np.asarray(label_index))
        label_perplexity = log_probs.gather(
            dim=0,
            index=tokens,
        )
        PERPLEXITY = label_perplexity.item()

        if len(ranking_position) >0 and ranking_position[0].shape[0] != 0:
            rank = ranking_position[0][0] + 1

            # print("rank: {}".format(rank))

            if rank >= 0:
                MRR = (1/rank)
            if rank >= 0 and rank <= P_AT:
                P_AT_X = 1.
            if rank == 1:
                P_AT_1 = 1.

    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY
    #
    # print("MRR: {}".format(experiment_result["MRR"]))
    # print("P_AT_X: {}".format(experiment_result["P_AT_X"]))
    # print("P_AT_1: {}".format(experiment_result["P_AT_1"]))
    # print("PERPLEXITY: {}".format(experiment_result["PERPLEXITY"]))

    return MRR, P_AT_X, experiment_result, return_msg


def __overlap_negation(index_max_probs__negated, index_max_probs):
    # compares first ranked prediction of affirmative and negated statements
    # if true 1, else: 0
    return int(index_max_probs__negated == index_max_probs)


def get_negation_metric(log_probs, masked_indices, log_probs_negated,
                        masked_indices_negated, vocab, index_list=None,
                        topk = 1):

    return_msg = ""
    # if negated sentence present
    if len(masked_indices_negated) > 0:

        log_probs, index_max_probs, _ = \
            __max_probs_values_indices(masked_indices, log_probs, topk=topk)
        log_probs_negated, index_max_probs_negated, _ = \
            __max_probs_values_indices(masked_indices_negated,
                                       log_probs_negated, topk=topk)

        # overlap btw. affirmative and negated first ranked prediction: 0 or 1
        overlap = __overlap_negation(index_max_probs_negated[0],
                                     index_max_probs[0])
        # rank corrl. btw. affirmative and negated predicted log_probs
        spearman_rank_corr = scipy.stats.spearmanr(log_probs,
                                                   log_probs_negated)[0]

    else:
        overlap = np.nan
        spearman_rank_corr = np.nan

    return overlap, spearman_rank_corr, return_msg
