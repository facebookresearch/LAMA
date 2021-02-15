# LAMA(LAnguage Model Analysis) を再確認してみた

2021-01-12
2021-01-29 更新
2021-02-15 更新

Yasuhiro MORIOKA

## 概要

LAMA(LAnguage Model Analysis) の環境をそのまま使って、結果の再現を試みた。
README.md の手順どおりに進めただけで、独自のデータセットやモデルで確認していない。
LAMA-UHN, Negated-LAMA は試しておきたい。

BERT, BERT-large, Elmo については Google-RE, T-REx でほぼ同様の結果を得た。Elmo-5Bは未確認。
Transformer-XL, GPTはおそらく実行環境のメモリサイズの問題で実行できていない。
メモリサイズの問題か ConceptNetの評価中に強制終了する。

RoBERTaは配布状態ではサポートされないが、動作させた。BERTより少し劣る結果。

GPT-2もサポートした。

## 内容

* 環境
    * ThinkPad E495 (AMD Ryzen5 2.1GHz, RAM 32GB, GPUなし)
    * Windows 10 Home, WSL2, Ubuntu 20.04
* 修正
    * Elmoモデルの状態クリアを追加
        * https://github.com/facebookresearch/LAMA/issues/30
        * 非効率だが relation 別の評価のたびに￥モデルをクリア。
    * GPT, FairSeq RoBERTa 向け pre-trained_language_models を定義
    * FairSeq/HuggingFace Roberta向けSpecial tokenを定義
    * Fairseq RoBERTa向けモデルダウンロード、vocaburaryのintersection取得を修正
    * HuggingFace RoBERTa向けモデルダウンロード, HfRobertaConnectorを作成
        * https://github.com/facebookresearch/LAMA/issues/15
        * bert_connectorを基本に gpt_connector のtokernizerの扱いを流用。
    * MRR, precision計算での len(list_of_results) == 0 の場合を処理
    * CUDAが利用できない場合の警告メッセージを抑制
    * GPT-2 モデルダウンロード、connectorを作成。

* 実行
    * "The LAMA probe" の手順をそのまま実行。

* 実行時の注意

    * pyenv-virtualenv 構成で minoconda-3.7 環境を用意し、さらに lama37環境を用意した。
    * pyenv と anaconda のactivateが衝突するので、次の手順で回避。
    * またローカルのlamaモジュールをロードできるようPYTHONPATHを修正。

    ```
    $ pyenv activate miniconda-3.7.0/envs/lama37
    $ export PYTHONPATH=.''
    $ python scripts/run_experiments.py 2>&1 | tee output.log
    ```

    * 参考
        * pyenvとanacondaを共存させる時のactivate衝突問題の回避策3種類 - Qiita
            * https://qiita.com/y__sama/items/f732bb7bec2bff355b69 
        * ModuleNotFoundError: No module named 'lama' · Issue #20 · facebookresearch/LAMA
            * https://github.com/facebookresearch/LAMA/issues/20


* 結果
    * BERT, BERT-large .. ほぼ論文のとおり。
    * Elmo .. Google-RE, T-REx の評価後、ConceptNetでの評価に移るが途中でエラーも警告も出力せずに終了する。
    * Elmo-5B .. 未実施
    * Transformer-XL .. RuntimeError: $ Torch: invalid memory size -- maybe an overflow? at /pytorch/aten/src/TH/THGeneral.cpp:188 エラー。 
    * GPT .. 大量の word FOO from vocab_subset in model vocabulary!　警告が表示され、評価回数が0となって div0 エラー。
    * Fairseq RoBERTa ..　BERTよりも少し悪い。ConceptNetの評価中にメモリ確保エラー。
        * https://github.com/facebookresearch/LAMA/issues/16
    * HuggingFace RoBERTa .. Fairseq RoBERTaと同様。
    * 各種エラーはメモリサイズさえ大きければ生じない印象。

* TODO
    * P27に対応する文テンプレートが T-REx では不適切 https://github.com/facebookresearch/LAMA/issues/40

* 疑問
    * vocab_intersection での除外条件が粗い印象。これしかないのかもしれないが。
        * nlp = spacy('en') に基づく。
        * stop_word は上記 nlp が返すもの
        * punctuation, symbolは各wordを nlp(word)の入力として tokenize結果のPOSで判断。よって数字だけでもPUNCTと判断されるケースがたびたび生じている。文脈が不足しているのだろう。
    * BPEを利用しているので、同じ語でも " word" と "word" の2つのトークンが対応する（はず）。前者は文中で登場するもの。後者は文頭に登場するか、語頭に続くものとして登場する。LAMAは1単語が複数トークンに分かれるケースを除外しているので、後者は語彙から除くことになる。しかし、文頭に登場する場合は前者と同様の意味を持つのではないか? obj-rel-subj のうちobjは文頭に出現することが多いだろうと考えると、そこで情報が失われていることはないのだろうか。まずはそこは捨てて考えてみているということか?

## 参考

* https://github.com/facebookresearch/LAMA
* https://arxiv.org/pdf/1909.01066.pdf
* https://openreview.net/forum?id=025X0zPfn

* https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md

* http://lotus.kuee.kyoto-u.ac.jp/~kurita/snlp2019_kurita.pdf
* https://blog.hoxo-m.com/entry/2019/10/24/083000#3-Language-Models-as-Knowledge-Bases
* https://twitter.com/gneubig/status/1177276621172150272

