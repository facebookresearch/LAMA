# LAMA: LAnguage Model Analysis
<img align="middle" src="img/logo.png" height="256" alt="LAMA">

LAMA is nothing more that a set of connectors to pre-trained language models. <br>
LAMA exposes a transparent and unique interface to use:

- Transformer-XL (Dai et al., 2019)
- BERT (Devlin et al., 2018)
- ELMo (Peters et al., 2018)
- GPT (Radford et al., 2018)
- fairseq (Dauphin et al., 2017)

Actually, LAMA is also a beautiful animal.

## What can you do with LAMA?

### 1. Encode a list of sentences
and use the vectors in your downstream task!

```bash
pip install -e git+https://github.com/facebookresearch/LAMA#egg=LAMA
```

```python
import argparse
from lama.build_encoded_dataset import encode, load_encoded_dataset

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
        ["The dog is sleeping on the sofa .", "He makes happy noises ."],  # two-sentence
        ]

encoded_dataset = encode(args, sentences)
print("Embedding shape: %s" % str(encoded_dataset[0].embedding.shape))
print("Tokens: %r" % encoded_dataset[0].tokens)

# save on disk the encoded dataset
encoded_dataset.save("test.pkl")

# load from disk the encoded dataset
new_encoded_dataset = load_encoded_dataset("test.pkl")
print("Embedding shape: %s" % str(new_encoded_dataset[0].embedding.shape))
print("Tokens: %r" % new_encoded_dataset[0].tokens)
```

### 2. Fill a sentence with a gap.

You should use the symbol ```[MASK]``` to specify the gap.
Only single-token gap supported - i.e., a single ```[MASK]```.
```bash
python lama/eval_generation.py  \
--lm "bert"  \
--t "The cat is on the [MASK]."
```
<img align="middle" src="img/cat_on_the_phone.png" height="470" alt="cat_on_the_phone">
<img align="middle" src="img/cat_on_the_phone.jpg" height="190" alt="cat_on_the_phone">
<sub><sup>source: https://commons.wikimedia.org/wiki/File:Bluebell_on_the_phone.jpg</sup></sub>


## Dependencies

(optional) It might be a good idea to use a separate conda environment. It can be created by running:
```
conda create -n lama36 python=3.6 && conda activate lama36
```

Clone the repo
```bash
git clone git@github.com:facebookresearch/LAMA.git && cd LAMA
```
Install as an editable package:
```bash
pip install --editable .
```

If you get an error in mac os x, please try running this instead
```bash
CFLAGS="-Wno-deprecated-declarations -std=c++11 -stdlib=libc++" pip install --editable .
```

Finally, install spacy model
```bash
python3 -m spacy download en
```

## Download the models

### DISCLAIMER: ~55 GB on disk
```bash
chmod +x download_models.sh
./download_models.sh
```
The script will create and populate a _pre-trained_language_models_ folder.
If you are interested in a particular model please edit the script.

## Language Model(s) options

Option to indicate which language model(s) to use:
* __--language-models/--lm__ : comma separated list of language models (__REQUIRED__)

### BERT
BERT pretrained models can be loaded both: (i) passing the name of the model and using huggingface cached versions or (ii) passing the folder containing the vocabulary and the PyTorch pretrained model (look at convert_tf_checkpoint_to_pytorch in [here](https://github.com/huggingface/pytorch-pretrained-BERT) to convert the TensorFlow model to PyTorch).

* __--bert-model-dir/--bmd__ : directory that contains the BERT pre-trained model and the vocabulary
* __--bert-model-name/--bmn__ : name of the huggingface cached versions of the BERT pre-trained model (default = 'bert-base-cased')
* __--bert-vocab-name/--bvn__ : name of vocabulary used to pre-train the BERT model (default = 'vocab.txt')

### ELMo

* __--elmo-model-dir/--emd__ : directory that contains the ELMo pre-trained model and the vocabulary (__REQUIRED__)
* __--elmo-model-name/--emn__ : name of the ELMo pre-trained model (default = 'elmo_2x4096_512_2048cnn_2xhighway')
* __--elmo-vocab-name/--evn__ : name of vocabulary used to pre-train the ELMo model (default = 'vocab-2016-09-10.txt')

### fairseq

* __--fairseq-model-dir/--fmd__ : directory that contains the fairseq pre-trained model and the vocabulary (__REQUIRED__)
* __--fairseq-model-name/--fmn__ : name of the fairseq pre-trained model (default = 'wiki103.pt')


### Transformer-XL

* __--transformerxl-model-dir/--tmd__ : directory that contains the pre-trained model and the vocabulary (__REQUIRED__)
* __--transformerxl-model-name/--tmn__ : name of the pre-trained model (default = 'transfo-xl-wt103')


### GPT

* __--gpt-model-dir/--gmd__ : directory that contains the gpt pre-trained model and the vocabulary (__REQUIRED__)
* __--gpt-model-name/--gmn__ : name of the gpt pre-trained model (default = 'openai-gpt')


## Evaluate Language Model(s) Generation

options:
* __--text/--t__ : text to compute the generation for
* __--i__ : interactive mode
one of the two is required

example considering both BERT and ELMo:
```bash
python lama/eval_generation.py \
--lm "bert,elmo" \
--bmd "pre-trained_language_models/bert/cased_L-24_H-1024_A-16/" \
--emd "pre-trained_language_models/elmo/original/" \
--t "The cat is on the [MASK]."
```

example considering only BERT with the default pre-trained model, in an interactive fashion:
```bash
python lamas/eval_generation.py  \
--lm "bert"  \
--i
```


## Get Contextual Embeddings

```bash
python lama/get_contextual_embeddings.py \
--lm "bert,elmo" \
--bmn bert-base-cased \
--emd "pre-trained_language_models/elmo/original/"
```


## References

- __(Dai et al., 2019)__ Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le, and Ruslan Salakhutdi. _Transformer-xl: Attentive language models beyond a fixed-length context_. CoRR, abs/1901.02860.

- __(Peters et al., 2018)__ Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. _Deep contextualized word representations_. NAACL-HLT 2018

- __(Devlin et al., 2018)__ Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. _BERT: pre-training of deep bidirectional transformers for language understanding_. CoRR, abs/1810.04805.

- __(Radford et al., 2018)__ Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. _Improving language understanding by generative pre-training_.

- __(Dauphin et al., 2017)__ Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. _Language modeling with gated convolutional networks_. ICML 2017
