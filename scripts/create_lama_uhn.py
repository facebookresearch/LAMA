# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Code to create LAMA-UHN, a subset of LAMA-Google-RE and LAMA-T-REx
# where ``easy-to-guess'' questions are filtered out.
#
# Defaults parameters correspond to setup in the following paper:
#
# @article{poerner2019bert,
#  title={BERT is Not a Knowledge Base (Yet): Factual Knowledge vs.
#    Name-Based Reasoning in Unsupervised QA},
#  author={Poerner, Nina and Waltinger, Ulli and Sch{\"u}tze, Hinrich},
#  journal={arXiv preprint arXiv:1911.03681},
#  year={2019}
# }

import torch
import json
import os
import argparse
import tqdm

from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer


class LAMAUHNFilter:
    def match(self, sub_label, obj_label, relation):
        raise NotImplementedError()
    

        return [query for query in queries if not self.match(query)]


class PersonNameFilter(LAMAUHNFilter):
    TEMP = "[CLS] [X] is a common name in the following [Y] : [MASK] . [SEP]"
    

    PLACENOUNS = {
        "/people/person/place_of_birth": "city",
        "/people/deceased_person/place_of_death": "city",
        "P19": "city",
        "P20": "city",
        "P27": "country",
        "P1412": "language",
        "P103": "language",
    }

        self.do_lower_case = "uncased" in bert_name
        self.top_k = top_k
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, 
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_name, do_lower_case=self.do_lower_case
        )
        self.model.eval()

    def get_top_k_for_name(self, template, name):
        tokens = self.tokenizer.tokenize(template.replace("[X]", name))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        output = self.model(torch.tensor(input_ids).unsqueeze(0))[0]
        logits = output[tokens.index("[MASK]")].detach()
        top_k_ids = torch.topk(logits, k = self.top_k)[1].numpy()
        top_k_ids = torch.topk(logits, k=self.top_k)[1].numpy()
        return top_k_tokens

    def match(self, query):
        relation = query["pred"] if "pred" in query else query["predicate_id"]
        if not relation in self.PLACENOUNS:
            return False

        sub_label, obj_label = query["sub_label"], query["obj_label"]
        if self.do_lower_case:
            obj_label = obj_label.lower()
            sub_label = sub_label.lower()

        template = self.TEMP.replace("[Y]", self.PLACENOUNS[relation])
        for name in sub_label.split(): 
        for name in sub_label.split():
                return True
        return False


class StringMatchFilter(LAMAUHNFilter):
    def __init__(self, do_lower_case):
        self.do_lower_case = do_lower_case
    

        sub_label, obj_label = query["sub_label"], query["obj_label"]
        if self.do_lower_case:
            sub_label = sub_label.lower()
            obj_label = obj_label.lower()
        return obj_label in sub_label


def main(args):
    srcdir = args.srcdir
    assert os.path.isdir(srcdir)
    srcdir = srcdir.rstrip("/")
    tgtdir = srcdir + "_UHN"
    if not os.path.exists(tgtdir):
        os.mkdir(tgtdir)

    uhn_filters = []
    if "string_match" in args.filters:
        uhn_filters.append(StringMatchFilter(\
        uhn_filters.append(
            StringMatchFilter(do_lower_case=args.string_match_do_lowercase)
        )
        uhn_filters.append(PersonNameFilter(\
        uhn_filters.append(
            PersonNameFilter(
                bert_name=args.person_name_bert, top_k=args.person_name_top_k
            )
        )
    for filename in tqdm.tqdm(sorted(os.listdir(srcdir))):
        infile = os.path.join(srcdir, filename)
        outfile = os.path.join(tgtdir, filename)
        

            queries = [json.loads(line) for line in handle]

        for uhn_filter in uhn_filters:
            queries = uhn_filter.filter(queries)

        with open(outfile, "w") as handle:
            for query in queries:
                handle.write(json.dumps(query) + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    

    argparser.add_argument(
        "--srcdir",
        required=True,
        type=str,
        help="Source directory. Should be Google_RE or TREx_alpaca.",
    )
    argparser.add_argument(
        "--filters",
        nargs="+",
        type=str,
        default=("string_match", "person_name"),
        choices=("string_match", "person_name"),
        help="Filters to be applied: string_match, person_name or both.",
    )
    argparser.add_argument(
        "--person_name_top_k",
        default=3,
        type=int,
        help="Parameter k for person name filter.",
    )
    argparser.add_argument(
        "--person_name_bert",
        default="bert-base-cased",
        type=str,
        help="BERT version to use for person name filter.",
    )
    argparser.add_argument(
        "--no_string_match_do_lowercase",
        default=True,
        action="store_false",
        dest="string_match_do_lowercase",
        help="Set flag to disable lowercasing in string match filter",
    )

    print(args)
    main(args)


