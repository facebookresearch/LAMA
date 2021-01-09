# LAMA(LAnguage Model Analysis) を再確認してみた

2021-01-09

Yasuhiro MORIOKA

## 概要

LAMA(LAnguage Model Analysis) の環境をそのまま使って、結果の再現を試みた。README.md の手順どおりに進めただけで、独自のデータセットやモデルで確認していない。

BERT, BERT-large, Elmoについては Google-RE, T-REx でほぼ同様の結果を得た。Elmo-5Bは未確認。

対応されているはずの Transofmer-XL, GPT, RoBERTa は、いずれも実行できなかった。

## 内容

* 環境
    * ThinkPad E495 (AMD Ryzen5 2.1GHz, RAM 32GB, GPUなし)
* 修正
    * Elmoモデルの状態クリアを追加
        * https://github.com/facebookresearch/LAMA/issues/30
    * GPT, RoBERTa 向け pre-trained_language_models を定義
    * CUDAが利用できない場合の警告メッセージを抑制
    
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
    * RoBERTa ..　モデルのロードに失敗。huggingface roberta-baseでなく pytorch/fairseq のモデルを使う必要があるのかもしれない。

## 参考

* https://github.com/facebookresearch/LAMA
* https://arxiv.org/pdf/1909.01066.pdf
* https://openreview.net/forum?id=025X0zPfn

* http://lotus.kuee.kyoto-u.ac.jp/~kurita/snlp2019_kurita.pdf
* https://blog.hoxo-m.com/entry/2019/10/24/083000#3-Language-Models-as-Knowledge-Bases
* https://twitter.com/gneubig/status/1177276621172150272

