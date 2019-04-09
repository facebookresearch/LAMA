# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import pytorch_pretrained_bert.tokenization as btok
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F


class CustomBaseTokenizer(BasicTokenizer):

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = btok.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:

            # pass MASK forward
            if MASK in token:
                split_tokens.append(MASK)
                if token != MASK:
                    remaining_chars = token.replace(MASK,"").strip()
                    if remaining_chars:
                        split_tokens.append(remaining_chars)
                continue

            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = btok.whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class Bert(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        bert_model_name = args.bert_model_name
        dict_file = bert_model_name

        if args.bert_model_dir is not None:
            # load bert model from file
            bert_model_name = str(args.bert_model_dir) + "/"
            dict_file = bert_model_name+args.bert_vocab_name
            self.dict_file = dict_file
            print("loading BERT model from {}".format(bert_model_name))
        else:
            # load bert model from huggingface cache
            pass

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(dict_file)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Add custom tokenizer to avoid splitting the ['MASK'] token
        custom_basic_tokenizer = CustomBaseTokenizer(do_lower_case = do_lower_case)
        self.tokenizer.basic_tokenizer = custom_basic_tokenizer

        # Load pre-trained model (weights)
        # ... to get prediction/generation
        self.masked_bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

        self.masked_bert_model.eval()

        # ... to get hidden states
        self.bert_model = self.masked_bert_model.bert

        self.pad_id = self.inverse_vocab[BERT_PAD]

        self.unk_index = self.inverse_vocab[BERT_UNK]

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # print(final_tokens_tensor)
        # print(final_segments_tensor)
        # print(final_attention_mask)
        # print(final_tokens_tensor.shape)
        # print(final_segments_tensor.shape)
        # print(final_attention_mask.shape)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(BERT_SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(BERT_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,BERT_CLS)
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK:
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_bert_model.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            log_probs = F.log_softmax(logits, dim=-1).cpu()

        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_encoder_layers, _ = self.bert_model(
                tokens_tensor.to(self._model_device),
                segments_tensor.to(self._model_device))

        all_encoder_layers = [layer.cpu() for layer in all_encoder_layers]

        sentence_lengths = [len(x) for x in tokenized_text_list]

        # all_encoder_layers: a list of the full sequences of encoded-hidden-states at the end
        # of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
        # encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
        return all_encoder_layers, sentence_lengths, tokenized_text_list
