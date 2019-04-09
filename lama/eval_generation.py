# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics


def main(args):

    if not args.text and not args.interactive:
        msg = "ERROR: either you start LAMA eval_generation with the " \
              "interactive option (--i) or you pass in input a piece of text (--t)"
        raise ValueError(msg)

    stopping_condition = True

    print("Language Models: {}".format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    while stopping_condition:
        if args.text:
            text = args.text
            stopping_condition = False
        else:
            text = input("insert text:")

        if args.split_sentence:
            import spacy
            # use spacy to tokenize input sentence
            nlp = spacy.load(args.spacy_model)
            tokens = nlp(text)
            print(tokens)
            sentences = []
            for s in tokens.sents:
                print(" - {}".format(s))
                sentences.append(s.text)
        else:
            sentences = [text]

        if len(sentences) > 2:
            print("WARNING: only the first two sentences in the text will be considered!")
            sentences = sentences[:2]

        for model_name, model in models.items():
            print("\n{}:".format(model_name))
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list)

            # prediction and perplexity for the whole softmax
            print_sentence_predictions(original_log_probs_list[0], token_ids, model.vocab, masked_indices=masked_indices)


if __name__ == '__main__':
    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    main(args)
