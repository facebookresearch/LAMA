# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from lama.modules.base_connector import *
from fairseq import tasks, utils
from fairseq.tokenizer import Tokenizer


def copy_args(args_from, args_to):
    args_to.task = args_from.task
    args_to.use_cuda = False
    args_to.data = args_from.data
    args_to.output_dictionary_size = args_from.output_dictionary_size


class Fairseq(Base_Connector):

    def __init__(self, args):
        super().__init__()

        fairseq_pretrained_model = str(args.data)+"/"+str(args.fairseq_model_name)
        pre_task = tasks.setup_task(args)
        print('| loading model from {}'.format(fairseq_pretrained_model))
        self.dict_file = fairseq_pretrained_model+"/dict.txt"
        models, self.model_args = utils.load_ensemble_for_inference([fairseq_pretrained_model], pre_task)

        copy_args(args,self.model_args)

        self.task = tasks.setup_task(self.model_args)
        self.model = models[0]

        self.decoder = self.model.decoder # <class 'fairseq.models.fconv.FConvDecoder'>

        # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
        self.model.make_generation_fast_()

        self.map_indices = None
        self.pad = self.task.target_dictionary.pad

        # add MASK to dictionary
        self.mask_id = self.task.dictionary.add_symbol(MASK)

        self.vocab = self.task.dictionary.symbols[:]

        # reinitialize inverse vocab
        self._init_inverse_vocab()
        self.unk_index = self.inverse_vocab[FAIRSEQ_UNK]

    def _cuda(self):
        self.model.cuda()

    def get_id(self, string):
        tokens = Tokenizer.tokenize(
            string, self.task.dictionary, add_if_not_exist=False, append_eos=False
        ).long()
        indexed_string = tokens.numpy()
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_text(self, sentences):
        text = " {} ".format(FAIRSEQ_EOS).join(sentences)
        return text

    def __get_decoder_out(self, input_sample):

        """Score a batch of translations."""
        sample = utils.move_to_cuda(input_sample) if self._model_device == 'cuda' else input_sample
        with torch.no_grad():
            self.model.eval()
            decoder_out = self.model.forward(src_tokens=sample["src_tokens"], src_lengths=sample["src_lengths"])
        return decoder_out

    def __create_sample(self, line):
        tokens = Tokenizer.tokenize(
            line, self.task.dictionary, tokenize=default_tokenizer, add_if_not_exist=False
        ).long()
        sample = {}
        # target is the sentence, for source, rotate item one token to the left (would start with eos)
        tokens_list = tokens.numpy()
        tokens_list = np.insert(tokens_list, 0, self.task.dictionary.eos())
        tokens = torch.from_numpy(tokens_list)
        sample["src_tokens"] = tokens.unsqueeze(0) # add a dimension for the batch
        sample["src_lengths"] = tokens.size()[0]
        sample['target'] = None # this will disable the efficient softmax approximation
        return sample

    def __create_sample_batch(self, line_list):
        tokens_list = []
        for line in line_list:
            tokens = Tokenizer.tokenize(
                line, self.task.dictionary, tokenize=default_tokenizer, add_if_not_exist=False
            ).tolist()
            tokens.insert(0, self.task.dictionary.eos()) # insert EOS at the beginning of the sentence
            tokens_list.append(tokens)

        src_lengths_list = [len(tokens) for tokens in tokens_list]
        token_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(t) for t in tokens_list],
            batch_first=True,
            padding_value=self.task.dictionary.eos())
        sample = {}
        sample["src_tokens"] = token_tensor
        sample["src_lengths"] = torch.LongTensor(src_lengths_list)
        sample['target'] = None # this will disable the efficient softmax approximation
        return sample

    def get_batch_generation(self, sentences_list, logger=None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        text_list = []
        for sentences in sentences_list:
            text_list.append(self.__get_text(sentences))

        sample = self.__create_sample_batch(text_list)

        if logger is not None:
            msg = ""
            for tokenized_text in sample["src_tokens"].cpu().numpy():
                msg += "{}\n".format([self.task.dictionary[idx] for idx in tokenized_text])
            logger.debug("\n{}\n".format(msg))


        # look for masked indices and substitute with UNK
        masked_indices_list = []
        for tokenized_text in sample['src_tokens']:
            masked_indices = []
            for i, token in enumerate(tokenized_text.tolist()):
                if (token == self.mask_id) and (i > 0):
                    masked_indices.append(i - 1)
                    tokenized_text[i] = self.unk_index # replace MASK with <unk>
            masked_indices_list.append(masked_indices)

        with torch.no_grad():
            decoder_out = self.__get_decoder_out(sample)
            batched_log_probs = self.decoder.get_normalized_probs(
                decoder_out, log_probs=True, sample=sample)
        log_probs = batched_log_probs.cpu()

        # Remove first symbols from tokens to store only "targets" and undo
        # padding.
        lengths = sample['src_lengths'].tolist()
        output_tokens_list = [
            sample['src_tokens'][i, 1:lengths[i]].cpu().numpy()
            for i in range(len(log_probs))
        ]
        # Remove last token from log_probs.
        log_probs = log_probs[:, :-1]

        return log_probs, output_tokens_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        text_list = []
        for sentences in sentences_list:
            text_list.append(self.__get_text(sentences))

        sample = self.__create_sample_batch(text_list)

        with torch.no_grad():
            decoder_out = self.__get_decoder_out(sample)

            #**decoder_out** (tuple): a tuple with two elements, where the
            # first element is the last decoder layer's output and the
            # second element is the same quantity summed with the input
            # embedding (used for attention). The shape of both tensors is
            # `(batch, src_len, embed_dim)`.
            output = decoder_out[0].cpu()

        tokenized_text_list = []
        for tokenized_text in sample["src_tokens"].cpu().numpy():
            tokenized_text_list.append([self.task.dictionary[idx] for idx in tokenized_text])

        sentence_lengths = [len(x) for x in tokenized_text_list]

        # fairseq returns only the last decoder layer's output, [] to have the same format as other models
        return [output], sentence_lengths, tokenized_text_list
