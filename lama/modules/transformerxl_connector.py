# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pytorch_pretrained_bert import TransfoXLLMHeadModel, TransfoXLTokenizer
import numpy as np
from lama.modules.base_connector import *


class TransformerXL(Base_Connector):

    UNK_SYMBOL = '<unk>'
    EOS_SYMBOL = '<eos>'

    def __init__(self, args):
        super().__init__()

        if args.transformerxl_model_dir is not None:
            model_name = args.transformerxl_model_dir
            dict_file = model_name
            print("Loading Transformer XL model from {}".format(model_name))
        else:
            model_name = args.transformerxl_model_name
            dict_file = model_name

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = TransfoXLTokenizer.from_pretrained(dict_file)

        self.vocab = list(self.tokenizer.idx2sym)
        self._init_inverse_vocab()
        self.eos_id = self.inverse_vocab[self.EOS_SYMBOL]
        self.unk_symbol = self.UNK_SYMBOL

        # Load pre-trained model (weights)
        self.model = TransfoXLLMHeadModel.from_pretrained(model_name)
        self.model.eval()
        print(self.model.config)

    def _cuda(self):
        self.model.cuda()

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_input_tensors(self, sentence_list):
        """Concatenates, tokenize and converts a sentences to model inputs.

        Args:
            sentence_list: A list of strings. The string may contain a special
            [MASK] token.

        Returns:
            A tuple (src_tensor, dst_tensor, masked_indices, tokenized_text).
                src_tensor: torch.LongTensor with shape (seq_len), the input to
                    the new without the last symbol and with EOS prepended.
                dst_tensor: torch.LongTensor with shape (seq_len).
                masked_indices: A list of indices of [MASK] in dst_tensor.
                tokenized_text: A list of token string.
            """
        # Split the sentence by [MASK] and tokenize the chunks independently.
        tokenized_text = []
        masked_indices = []
        for sentence_idx, sentence in enumerate(sentence_list):
            if sentence_idx > 0:
                tokenized_text.append(self.EOS_SYMBOL)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text))
                    tokenized_text.append(self.unk_symbol)
                chunk = chunk.strip()
                if chunk:
                    tokenized_text.extend(self.tokenizer.tokenize(chunk))

        full_indexed_tokens = [
            self.eos_id
        ] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]

        return src_tensor, dst_tensor, masked_indices, tokenized_text

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        with torch.no_grad():
            log_probs, _ = self.model(src_tensor_batch.to(self._model_device))
            log_probs = log_probs.cpu()

        token_ids_list = [
            np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list
        ]

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, batched_sentence_list):
        batch = []
        for sentence_list in batched_sentence_list:
            tokenized_text = [self.eos_id]
            for sentence in sentence_list:
                tokenized_text.extend(self.tokenizer.tokenize(sentence))
                tokenized_text.append(self.EOS_SYMBOL)

            full_indexed_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_text))
            batch.append(full_indexed_tokens)

        tensor_batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True).to(self._model_device)

        last_hidden_state, _ = self.model.transformer(tensor_batch)

        last_hidden_state = last_hidden_state[:, 1:]

        # TODO
        sentence_lengths = None
        tokenized_text_list = None

        # As we only return the last layer, wrap it into a list
        return [last_hidden_state], sentence_lengths, tokenized_text_list
