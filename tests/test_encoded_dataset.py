# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from lama.build_encoded_dataset import encode, load_encoded_dataset
import os

if __name__ == '__main__':

    PARAMETERS= {
            "lm": "bert",
            "bert_model_name": "bert-large-cased",
            "bert_model_dir":
            "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32
            }

    args = argparse.Namespace(**PARAMETERS)

    sentences = [
            ["The cat is on the table ."],  # single-sentence instance
            ["The dog is sleeping on the sofa .", "He makes happy noises ."],  # two-sentences
            ]

    encoded_dataset = encode(args, sentences)
    print("Embedding shape: %s" % str(encoded_dataset[0].embedding.shape))
    print("Tokens: %r" % encoded_dataset[0].tokens)
    assert(encoded_dataset[0].embedding.shape[0] == 9)
    assert (encoded_dataset[0].embedding.shape[1] == 1024)
    encoded_dataset.save("test.pkl")

    new_encoded_dataset = load_encoded_dataset("test.pkl")
    print("Embedding shape: %s" % str(new_encoded_dataset[0].embedding.shape))
    print("Tokens: %r" % new_encoded_dataset[0].tokens)
    assert (encoded_dataset[0].embedding.shape[0] == 9)
    assert (encoded_dataset[0].embedding.shape[1] == 1024)

    os.remove("test.pkl")
    print("test successfully passed!")
