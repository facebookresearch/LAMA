# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse


def get_general_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language-models",
        "--lm",
        dest="models",
        help="comma separated list of language models",
        required=True,
    )
    parser.add_argument(
        "--spacy_model",
        "--sm",
        dest="spacy_model",
        default="en_core_web_sm",
        help="spacy model file path",
    )
    parser.add_argument(
        "--common-vocab-filename",
        "--cvf",
        dest="common_vocab_filename",
        help="common vocabulary filename",
    )
    parser.add_argument(
        "--interactive",
        "--i",
        dest="interactive",
        action="store_true",
        help="perform the evaluation interactively",
    )
    __add_bert_args(parser)
    __add_elmo_args(parser)
    __add_gpt_args(parser)
    __add_transformerxl_args(parser)
    __add_roberta_args(parser)
    return parser


def get_eval_generation_parser():
    parser = get_general_parser()
    parser.add_argument(
        "--text", "--t", dest="text", help="text to compute the generation for"
    )
    parser.add_argument(
        "--split_sentence",
        dest="split_sentence",
        action="store_true",
        help="split the input text in sentences",
    )
    return parser


def get_eval_KB_completion_parser():
    parser = get_general_parser()
    parser.add_argument(
        "--dataset-filename",
        "--df",
        dest="dataset_filename",
        help="filename containing dataset",
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        default="../experiments_logs/",
        help="logging directory",
    )
    parser.add_argument(
        "--full-logdir",
        help="Full path to the logging folder. If set, wiill override log_dir.",
    )
    parser.add_argument(
        "--template", dest="template", default="", help="template for surface relation"
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=32, help="batch size"
    )
    parser.add_argument(
        "--max-sentence-length",
        dest="max_sentence_length",
        type=int,
        default=100,
        help="max sentence lenght",
    )
    parser.add_argument(
        "--lowercase",
        "--lower",
        dest="lowercase",
        action="store_true",
        help="perform the evaluation using lowercase text",
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        type=int,
        default=-1,
        help="number of threads for evaluation metrics computation (defaults: all available)",
    )
    return parser


def __add_bert_args(parser):
    group = parser.add_argument_group("BERT")
    group.add_argument(
        "--bert-model-dir",
        "--bmd",
        dest="bert_model_dir",
        help="directory that contains the BERT pre-trained model and the vocabulary",
    )
    group.add_argument(
        "--bert-model-name",
        "--bmn",
        dest="bert_model_name",
        default="bert-base-cased",
        help="name of the BERT pre-trained model (default = 'bert-base-cased')",
    )
    group.add_argument(
        "--bert-vocab-name",
        "--bvn",
        dest="bert_vocab_name",
        default="vocab.txt",
        help="name of vocabulary used to pre-train the BERT model (default = 'vocab.txt')",
    )
    return group


def __add_roberta_args(parser):
    group = parser.add_argument_group("RoBERTa")
    group.add_argument(
        "--roberta-model-dir",
        "--rmd",
        dest="roberta_model_dir",
        help="directory that contains the ROBERTA pre-trained model and the vocabulary",
    )
    group.add_argument(
        "--roberta-model-name",
        "--rmn",
        dest="roberta_model_name",
        default="model.pt",
        help="name of the ROBERTA pre-trained model (default = 'model.pt')",
    )
    group.add_argument(
        "--roberta-vocab-name",
        "--rvn",
        dest="roberta_vocab_name",
        default="dict.txt",
        help="name of vocabulary used to pre-train the ROBERTA model (default = 'vocab.txt')",
    )
    return group


def __add_gpt_args(parser):
    group = parser.add_argument_group("GPT")
    group.add_argument(
        "--gpt-model-dir",
        "--gmd",
        dest="gpt_model_dir",
        help="directory that contains the gpt pre-trained model and the vocabulary",
    )
    group.add_argument(
        "--gpt-model-name",
        "--gmn",
        dest="gpt_model_name",
        default="openai-gpt",
        help="name of the gpt pre-trained model (default = 'openai-gpt')",
    )
    return group


def __add_transformerxl_args(parser):
    group = parser.add_argument_group("GPT")
    group.add_argument(
        "--transformerxl-model-dir",
        "--tmd",
        help="directory that contains the pre-trained model and the vocabulary",
    )
    group.add_argument(
        "--transformerxl-model-name",
        "--tmn",
        default="transfo-xl-wt103",
        help="name of the pre-trained model (default = 'transfo-xl-wt103')",
    )
    return group


def __add_elmo_args(parser):
    group = parser.add_argument_group("ELMo")
    group.add_argument(
        "--elmo-model-dir",
        "--emd",
        dest="elmo_model_dir",
        help="directory that contains the ELMo pre-trained model and the vocabulary",
    )
    group.add_argument(
        "--elmo-model-name",
        "--emn",
        dest="elmo_model_name",
        default="elmo_2x4096_512_2048cnn_2xhighway",
        help="name of the ELMo pre-trained model (default = 'elmo_2x4096_512_2048cnn_2xhighway')",
    )
    group.add_argument(
        "--elmo-vocab-name",
        "--evn",
        dest="elmo_vocab_name",
        default="vocab-2016-09-10.txt",
        help="name of vocabulary used to pre-train the ELMo model (default = 'vocab-2016-09-10.txt')",
    )
    group.add_argument(
        "--elmo-warm-up-cycles",
        dest="elmo_warm_up_cycles",
        type=int,
        default=5,
        help="ELMo warm up cycles",
    )
    return group


def parse_args(parser):
    args = parser.parse_args()
    args.models_names = [x.strip().lower() for x in args.models.split(",")]
    if "fconv" in args.models_names:
        if args.data is None:
            raise ValueError(
                "to use fconv you should specify the directory that contains "
                "the pre-trained model and the vocabulary with the option --fconv-model-dir/--fmd\n"
                "you can also specify the fconv model name with the option --fconv-model-name/--fmn (default = 'wiki103.pt')\n"
                "the vocabulary should be in the provided fconv-model-dir and be named dict.txt"
            )
    if "bert" in args.models_names:
        # use the default shortcut name of a Google AI's pre-trained model (default = 'bert-base-cased')
        pass
    if "elmo" in args.models_names:
        if args.elmo_model_dir is None:
            raise ValueError(
                "to use elmo you should specify the directory that contains "
                "the pre-trained model and the vocabulary with the option --elmo-model-dir/--emd\n"
                "you can also specify the elmo model name with the option --elmo-model-name/--emn (default = 'elmo_2x4096_512_2048cnn_2xhighway')\n"
                "and the elmo vocabulary name with the option --elmo-vocab-name/--evn (default = 'vocab-2016-09-10.txt')"
            )

    return args
