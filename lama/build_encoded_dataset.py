# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle as pkl
from tqdm import tqdm
try:
    import ujson as json
except ImportError:
    import json
import collections
import torch
from lama.modules import build_model_by_name

# A tuple containing a single example from the input dataset with sentences
# mapped into a sequence of vectors:
#   embeddings: tensor with shape (some_length, embedding_dim).
# Note that some_length differs from example to example, while
# embedding_dim is the same for all examples for the encoded dataset.
EncodedSentence = collections.namedtuple('EncodedSentence',
                                         'embedding, length, tokens')


class EncodedDataset(torch.utils.data.Dataset):

    def __init__(self, encoded_sentences=None):
        if encoded_sentences:
            # make sure encoded_sentences is a list of Strings
            assert isinstance(encoded_sentences, list)
            sample = encoded_sentences[0]
            assert len(sample) == 3
            assert isinstance(sample[0], torch.Tensor)
            self._encodings = encoded_sentences
        else:
            self._embeddings = []

    def __len__(self):
        return len(self._encodings)

    def __getitem__(self, idx):
        encoding = self._encodings[idx]
        embedding, sent_length, tokens = encoding

        return EncodedSentence(embedding=embedding, length=sent_length, tokens=tokens)

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self._encodings, f)

    def load(self, path):
        """ Read precomputed contextual embeddings from file

        :param path: path to the embedding file (in npz format)
        """
        with open(path, 'rb') as f:
            self._encodings = pkl.load(f)


def load_encoded_dataset(path):
    dataset = EncodedDataset()
    dataset.load(path)
    return dataset


def _batchify(sentences, batch_size):
    start = 0
    while start < len(sentences):
        yield sentences[start:start + batch_size]
        start += batch_size


def _aggregate_layers(embeddings):
    """ Average over all layers """
    new_embed = torch.stack(embeddings, 0)  # [#layers, #batchsize, #max_sent_len, #dim]
    agg_embed = torch.mean(new_embed, 0)  # [#batchsize, #max_sent_len, #dim]
    return agg_embed


def encode(args, sentences, sort_input=False):
    """Create an EncodedDataset from a list of sentences

    Parameters:
    sentences (list[list[string]]): list of elements. Each element is a list
                                    that contains either a single sentence
                                    or two sentences
    sort_input (bool): if true, sort sentences by number of tokens in them

    Returns:
    dataset (EncodedDataset): an object that contains the contextual
                              representations of the input sentences
    """
    print("Language Models: {}".format(args.lm))
    model = build_model_by_name(args.lm, args)

    # sort sentences by number of tokens in them to make sure that in all
    # batches there are sentence with a similar numbers of tokens
    if sort_input:
        sorted(sentences, key=lambda k: len(" ".join(k).split()) )

    encoded_sents = []
    for current_batch in tqdm(_batchify(sentences, args.batch_size)):
        embeddings, sent_lens, tokenized_sents = model.get_contextual_embeddings(current_batch)

        agg_embeddings = _aggregate_layers(embeddings)  # [#batchsize, #max_sent_len, #dim]
        sent_embeddings = [agg_embeddings[i, :l] for i, l in enumerate(sent_lens)]
        encoded_sents.extend(list(zip(sent_embeddings, sent_lens, tokenized_sents)))

    dataset = EncodedDataset(encoded_sents)
    return dataset