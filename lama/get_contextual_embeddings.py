# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
import lama.options as options

def main(args):
    sentences = [
        ["the cat is on the table ."],  # single-sentence instance
        ["the dog is sleeping on the sofa .", "he makes happy noises ."],  # two-sentence
    ]

    print("Language Models: {}".format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    for model_name, model in models.items():
        print("\n{}:".format(model_name))
        if args.cuda:
            model.try_cuda()
        contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
            sentences)

        # contextual_embeddings is a list of tensors, one tensor for each layer.
        # Each element contains one layer of the representations with shape
        # (x, y, z).
        #   x    - the batch size
        #   y    - the sequence length of the batch
        #   z    - the length of each layer vector

        print(f'Number of layers: {len(contextual_embeddings)}')
        for layer_id, layer in enumerate(contextual_embeddings):
            print(f'Layer {layer_id} has shape: {layer.shape}')

        print("sentence_lengths: {}".format(sentence_lengths))
        print("tokenized_text_list: {}".format(tokenized_text_list))


if __name__ == '__main__':
    parser = options.get_general_parser()
    parser.add_argument('--cuda', action='store_true', help='Try to run on GPU')
    args = options.parse_args(parser)
    main(args)
