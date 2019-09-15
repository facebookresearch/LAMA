#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"


echo "lowercase models"
echo "OpenAI GPT"
if [[ ! -f gpt/openai-gpt/config.json ]]; then
  rm -rf 'gpt/openai-gpt'
  mkdir -p 'gpt/openai-gpt'
  cd 'gpt/openai-gpt'
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json' -O vocab.json
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt' -O merges.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json' -O 'config.json'
  cd ../..
fi

echo "BERT BASE LOWERCASED"
if [[ ! -f bert/uncased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
  unzip uncased_L-12_H-768_A-12.zip
  rm uncased_L-12_H-768_A-12.zip
  cd uncased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
  tar -xzf bert-base-uncased.tar.gz
  rm bert-base-uncased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT LARGE LOWERCASED"
if [[ ! -f bert/uncased_L-24_H-1024_A-16/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
  unzip uncased_L-24_H-1024_A-16.zip
  rm uncased_L-24_H-1024_A-16.zip
  cd uncased_L-24_H-1024_A-16
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz"
  tar -xzf bert-large-uncased.tar.gz
  rm bert-large-uncased.tar.gz
  rm bert_model*
  cd ../../
fi


echo 'cased models'
echo 'Transformer XL'
if [[ ! -f 'transformerxl/transfo-xl-wt103/config.json' ]]; then
  rm -rf 'transformerxl/transfo-xl-wt103'
  mkdir -p 'transformerxl/transfo-xl-wt103'
  cd 'transformerxl/transfo-xl-wt103'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-vocab.bin' -O 'vocab.bin'
  # Extracting plain text vocab for debugging purposes.
  python -c 'import torch; print(*torch.load("vocab.bin")["sym2idx"].keys(), sep="\n")' | sort > vocab.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json' -O 'config.json'
  cd ../../
fi

echo "ELMO ORIGINAL 5.5B"
if [[ ! -f elmo/original5.5B/vocab-enwiki-news-500000.txt ]]; then
  mkdir -p 'elmo'
  cd elmo
  mkdir -p 'original5.5B'
  cd original5.5B
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_softmax_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/tf_checkpoint/vocab-enwiki-news-500000.txt"
  cd ../../
fi

echo "ELMO ORIGINAL"
if [[ ! -f elmo/original/vocab-2016-09-10.txt ]]; then
  mkdir -p 'elmo'
  cd elmo
  mkdir -p 'original'
  cd original
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt"
  cd ../../
fi


echo "BERT BASE CASED"
if [[ ! -f bert/cased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
  unzip cased_L-12_H-768_A-12
  rm cased_L-12_H-768_A-12.zip
  cd cased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
  tar -xzf bert-base-cased.tar.gz
  rm bert-base-cased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT LARGE CASED"
if [[ ! -f bert/cased_L-24_H-1024_A-16/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
  unzip cased_L-24_H-1024_A-16.zip
  rm cased_L-24_H-1024_A-16.zip
  cd cased_L-24_H-1024_A-16
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz"
  tar -xzf bert-large-cased.tar.gz
  rm bert-large-cased.tar.gz
  rm bert_model*
  cd ../../
fi


cd "$ROOD_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_cased.txt" ]; then
  python lama/vocab_intersection.py
else
  echo 'Already exists. Run to re-build:'
  echo 'python util_KB_completion.py'
fi

