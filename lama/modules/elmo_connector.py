# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import torch
import h5py
from allennlp.modules.elmo import _ElmoBiLm #, Elmo as AllenNLP_Elmo
from allennlp.modules.elmo import batch_to_ids
import numpy as np
from lama.modules.base_connector import *


def get_text(sentences):
    text = " {} {} ".format(ELMO_END_SENTENCE, ELMO_START_SENTENCE).join(sentences)
    return default_tokenizer(text)


class Elmo(Base_Connector):

    def __init__(self, args):
        super().__init__()

        options_file = args.elmo_model_dir + "/" + args.elmo_model_name + "_options.json"
        weight_file  = args.elmo_model_dir + "/" + args.elmo_model_name + "_weights.hdf5"
        with open(options_file, encoding='utf-8') as data_file:
            data = json.loads(data_file.read())
        self.hidden_size = data['lstm']['projection_dim']

        # 1. Vocabulary
        vocab_to_cache = None

        # use vocabulary that was used for training to initialize top layer
        self.softmax_file = args.elmo_model_dir + "/" + args.elmo_model_name + "_softmax_weights.hdf5"
        self.dict_file = args.elmo_model_dir + "/" + args.elmo_vocab_name
        self.__init_vocab(self.dict_file)

        # Note The Elmo class in allennlp.modules.elmo will weight together all
        # of the layers and when initialized will just average them together.
        # _ElmoBiLm is used to have access to all the layers.

        # 2. ELMo model
        self.elmo_lstm = _ElmoBiLm(
            options_file = options_file,
            weight_file = weight_file,
            vocab_to_cache=vocab_to_cache
        )

        # 3. Top Layer
        # use pre-trained top layer
        self.__init_top_layer(softmax_file = self.softmax_file)

        self.unk_index = self.inverse_vocab[ELMO_UNK]
        
        self.warm_up_cycles = args.elmo_warm_up_cycles

    def __init_vocab(self, dict_file):
        with open(dict_file, "r") as f:
            lines = f.readlines()
        self.vocab = [x.strip() for x in lines]
        self._init_inverse_vocab()

    def __init_top_layer(self, softmax_file = None):
        with h5py.File(softmax_file, 'r') as fin:
            output_weights = fin['softmax']['W'][...]
            output_bias = fin['softmax']['b'][...]
        if (output_weights.shape[0] != len(self.vocab)):
            print("output_weights.shape[0] : {} != len(self.vocab) : {}".format(output_weights.shape[0],len(self.vocab)))
            indices = []
            for word in self.vocab:
                if word in self.inverse_vocab:
                    indices.append(self.inverse_vocab[word])
                else:
                    raise ValueError("word: {} not in original ELMo vocab".format(word))
            output_weights = np.take(output_weights, indices, axis=0)
            output_bias = np.take(output_bias, indices, axis=0)
        self.output_layer = torch.nn.Linear(self.hidden_size, len(self.vocab), bias=True)
        self.output_layer.weight = torch.nn.Parameter(torch.from_numpy(output_weights))
        self.output_layer.bias = torch.nn.Parameter(torch.from_numpy(output_bias))

    def optimize_top_layer(self, vocab_subset):

        for symbol in SPECIAL_SYMBOLS:
            if symbol in self.vocab and symbol not in vocab_subset:
                vocab_subset.append(symbol)

        # use given vocabulary for ELMo
        self.vocab = [ x for x in vocab_subset if x in self.inverse_vocab and x != ELMO_UNK ]

        self.__init_top_layer(softmax_file = self.softmax_file)
        
        # the inverse vocab initialization should be done after __init_top_layer
        self._init_inverse_vocab()
        

    def __get_tokend_ids(self, text):
        token_ids = [self.inverse_vocab[ELMO_START_SENTENCE]]
        for word in text.split():
            if word in self.inverse_vocab:
                idx = self.inverse_vocab[word]
            else:
                idx = self.unk_index #self.inverse_vocab[ELMO_UNK]
            token_ids.append(idx)
        token_ids.append(self.inverse_vocab[ELMO_END_SENTENCE])
        return np.array(token_ids)

    def get_id(self, string):
        token_ids = []
        for word in string.split():
            if word in self.inverse_vocab:
                idx = self.inverse_vocab[word]
            else:
                return None
            token_ids.append(idx)
        return token_ids

    def _cuda(self):
        """Move model to GPU."""
        self.elmo_lstm.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokenized_text_list = []
        for sentences in sentences_list:
            tokenized_text_list.append(get_text(sentences))

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        # look for masked indices
        masked_indices_list = []
        for tokenized_text in tokenized_text_list:
            masked_indices = []
            for i in range(len(tokenized_text)):
                token = tokenized_text[i]
                if (token == MASK):
                    masked_indices.append(i+1) # to align with the next shift
                    tokenized_text[i] = ELMO_UNK # replace MASK with <unk>
            masked_indices_list.append(masked_indices)

        character_ids = batch_to_ids(tokenized_text_list)
        batch_size = character_ids.shape[0]

        with torch.no_grad():
            
            bilm_input = character_ids.to(self._model_device)
            bilm_output = None
            for _ in range(self.warm_up_cycles):
                '''After loading the pre-trained model, the first few batches will be negatively 
                impacted until the biLM can reset its internal states. 
                You may want to run a few batches through the model to warm up the states before making 
                predictions (although we have not worried about this issue in practice).'''
                bilm_output = self.elmo_lstm(bilm_input)
            
            elmo_activations = bilm_output['activations'][-1].cpu() # last layer

            forward_sequence_output,backward_sequence_output = torch.split(elmo_activations, int(self.hidden_size), dim=-1)

            logits_forward = self.output_layer(forward_sequence_output)
            logits_backward = self.output_layer(backward_sequence_output)

            log_softmax = torch.nn.LogSoftmax(dim=-1)
            log_probs_forward = log_softmax(logits_forward)
            log_probs_backward = log_softmax(logits_backward)

        pad = torch.zeros([batch_size, 1, len(self.vocab)], dtype=torch.float)

        log_probs_forward_splitted = torch.split(log_probs_forward, 1, dim=1)
        log_probs_backward_splitted = torch.split(log_probs_backward, 1, dim=1)

        log_probs_forward = torch.cat(list([pad])+list(log_probs_forward_splitted[:-1]), dim=1) # shift forward +1 log_probs_forward
        log_probs_backward = torch.cat(list(log_probs_backward_splitted[1:])+list([pad]), dim=1) # shift backward -1 log_probs_backward

        avg_log_probs = (log_probs_forward + log_probs_backward) / 2

        num_tokens = avg_log_probs.shape[1]

        token_ids_list = []
        for tokenized_text in tokenized_text_list:
            token_ids = self.__get_tokend_ids(" ".join(tokenized_text).strip())
            while len(token_ids) < num_tokens:
                token_ids = np.append(token_ids, self.inverse_vocab[ELMO_END_SENTENCE])
            token_ids_list.append(token_ids)

        return avg_log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokenized_text_list = []
        for sentences in sentences_list:
            tokenized_text_list.append(get_text(sentences))
        character_ids = batch_to_ids(tokenized_text_list)

        with torch.no_grad():
            bilm_output = self.elmo_lstm(character_ids.to(self._model_device))
            activations = [act.cpu() for act in bilm_output['activations']]

        sentence_lengths = [len(x) for x in tokenized_text_list]

        return activations, sentence_lengths, tokenized_text_list
