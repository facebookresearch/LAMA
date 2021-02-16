# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from lama.modules.base_connector import *


class GPT2(Base_Connector):

    def __init__(self, args):
        super().__init__()

        if args.gpt2_model_dir is not None:
            # load GPT2 model from file
            gpt_model_name = str(args.gpt2_model_dir) + "/"
            dict_file = gpt_model_name
            print("loading GPT2 model from {}".format(gpt_model_name))
        else:
            # load GPT2 model from huggingface cache
            gpt_model_name = args.gpt2_model_name
            dict_file = gpt_model_name

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(dict_file)

        # GPT uses different way to represent BPE then BERT. Namely, the
        # final suffixes are indicated with </w> suffix, while pieces that must
        # be followed are written as is. In BERT the prefixes are written as is
        # while the parts that must follow (not be followed!) have '##' prefix.
        # There is no one-to-one coversion. But at least we may make pieces that
        # may form a full word look the same.
        # Note that we should be very careful now,
        # tokenizer.convert_tokens_to_ids won't work with our vocabulary.
            
        def convert_word(word):
            if word == GPT2_EOS:
                return word

            if word.startswith('Ġ'):  # the token starts with a whitespace
                return word[1:]

            return f'_{word}_'  # the token not start with a white space.
                                # may be not a head of a word,
                                # or may be a head of a sentence.

        _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items()))
        self.vocab = [convert_word(word) for word in gpt_vocab]
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.gpt_model.eval()
        # print(self.gpt_model.config)

        # Sanity check.
        assert len(self.vocab) == self.gpt_model.config.vocab_size
        #assert 0 == self.gpt_model.config.n_special

        self.eos_id = self.gpt_model.config.eos_token_id
        self.pad_id = self.gpt_model.config.eos_token_id
        self.unk_id = self.gpt_model.config.eos_token_id
        self.bos_id = self.gpt_model.config.bos_token_id
        self.model_vocab = self.vocab

    def _cuda(self):
        self.gpt_model.cuda()

    def get_id(self, string):
        indexed_string = self.tokenizer.encode(f'a {string}')[1:]
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
                tokenized_text.append(self.eos_id)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text))
                    tokenized_text.append(self.unk_id)  # use UNK as [MASK]
                chunk = chunk.strip()
                if chunk:
                    tokenized_sentence = self.tokenizer.encode(chunk)
                    tokenized_text.extend(tokenized_sentence)

        full_indexed_tokens = [
            self.bos_id
        ] + tokenized_text
        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]

        tokenized_text = self.tokenizer.decode(tokenized_text)

        return src_tensor, dst_tensor, masked_indices, tokenized_text

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        # The model uses shared embedding space for tokens and positions. More
        # precisely, the first len(vocab) indidices are reseved for words, the
        # last n_special symbols are reserved for special symbols and the rest
        # is used for positions. Softmax and embedding matrices are shared and
        # as result some of output "symbols" correspond to positions. To fix
        # that we have to manually remove logits for positions.
        with torch.no_grad():
            logits = self.gpt_model(src_tensor_batch.to(self._model_device))[0]
            logits = logits[..., :self.gpt_model.config.vocab_size]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        token_ids_list = [
            np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list
        ]

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        if try_cuda:
            self.try_cuda()

        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        with torch.no_grad():
            output = self.gpt_model.transformer(src_tensor_batch.to(self._model_device))

        # TODO
        sentence_lengths = None
        tokenized_text_list = None

        # As we only return the last layer, [] to have the same format as other models
        return [output], sentence_lengths, tokenized_text_list


