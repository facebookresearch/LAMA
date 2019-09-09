# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from colorama import init
from termcolor import colored
import numpy as np
import lama.modules.base_connector as base


def __exclude_tokens(token_ids, vocab):
    indices_to_exclude = []
    for i, tok in enumerate(token_ids):
        word_form = vocab[tok]
        if (word_form in base.SPECIAL_SYMBOLS):
            indices_to_exclude.append(i)
    return indices_to_exclude


def __print_generation(positional_scores, token_ids, vocab, rank_dict,
                       index_max_probs, value_max_probs, topk,
                       indices_to_exclude, masked_indices, print_on_console):
    init()  # colorful output
    msg = ""
    dash = '-' * 82
    msg += dash + "\n"
    msg += '{:<8s}{:<20s}{:<12s}{:<20}{:<12s}{:<12s}'.format(
                    "index", "token", "log_prob", "prediction",
                    "log_prob", "rank@{}".format(topk))
    msg += "\n" + dash
    if print_on_console:
        print(msg)
    msg += '\n'

    for idx, tok in enumerate(token_ids):

        word_form = vocab[tok]

        rank = -1
        if idx in rank_dict:
            rank = rank_dict[idx]
        index_max_prob = index_max_probs[idx]

        predicted_token_id = index_max_prob[0]

        value_max_prob = value_max_probs[idx]
        string_to_print = '{:<8d}{:<20s}{:<12.3f}{:<20s}{:<12.3f}{:<12d}'.format(
            idx,
            str(word_form),
            positional_scores[idx],
            str(vocab[predicted_token_id]),
            value_max_prob[0],
            rank
        )

        if print_on_console:
            if masked_indices is not None and idx in masked_indices:
                print(colored(string_to_print, 'grey', 'on_yellow'))
            elif indices_to_exclude is not None and idx in indices_to_exclude:
                print(colored(string_to_print, 'grey', 'on_grey'))
            else:
                print(string_to_print)
        msg += string_to_print + "\n"

    return msg


def __get_topk(log_probs, topk):
    value_max_probs, index_max_probs = torch.topk(input=log_probs, k=topk, dim=1)
    index_max_probs = index_max_probs.numpy()
    value_max_probs = value_max_probs.detach().numpy()
    return value_max_probs, index_max_probs


def print_sentence_predictions(log_probs, token_ids, vocab,
                               masked_indices=None, print_generation=True,
                               topk=1000):

    msg = "\n"
    log_probs = log_probs[:len(token_ids)]
    value_max_probs, index_max_probs = __get_topk(log_probs, topk)

    # remove special symbols from token_ids
    excluded_indices = __exclude_tokens([t for t in token_ids], vocab)

    # score only first mask
    masked_indices = masked_indices[:1]

    tokens = torch.from_numpy(np.asarray(token_ids))

    # get ranking position in topk
    query = tokens.squeeze().data.unsqueeze(-1)
    query = query.repeat(1, topk)

    ranking_position = (index_max_probs == query.numpy()).nonzero()

    rank_dict = dict(zip(*ranking_position))

    # get positional score of the correct token
    token_probs = log_probs.gather(
        dim=1,
        index=tokens.view(-1, 1),
    )
    positional_scores = token_probs.squeeze(-1).detach().numpy()

    score_sum = 0.
    count = 0
    for idx, score in enumerate(positional_scores):
        if idx not in excluded_indices:
            score_sum += score
            count += 1

    if count > 0:
        avg_nll_loss = - (score_sum / count)
    else:
        avg_nll_loss = 0.0
    perplexity = np.exp(avg_nll_loss)

    # print("positional_scores: {}".format(positional_scores))
    # print("avg_nll_loss: {}".format(avg_nll_loss))

    __print_generation(positional_scores, token_ids, vocab, rank_dict,
                       index_max_probs, value_max_probs, topk,
                       excluded_indices, masked_indices, print_generation)

    # msg += return_msg
    msg += '| Perplexity: {:.3f}\n'.format(perplexity)

    if print_generation:
        print("\n"+msg+"\n")

    return perplexity, msg


def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab
