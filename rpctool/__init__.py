#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Tools for Retail-Product-Checkout-Dataset(RPC)
'''
__version__ = "0.1.0"
__short_description__ = "Tools for Retail-Product-Checkout-Dataset(RPC)"
__author__ = "DIYer22, Cui Quan, "
__author_email__ = "yanglei@megvii.com; cuiquan@megvii.com"
__github_url__ = "https://github.com/DIYer22/retail_product_checkout_tools"
__support__ = "https://github.com/DIYer22/retail_product_checkout_tools/issues"

import boxx
import pandas as pd

from rpc_config import config
config.debug = True

def get_skudf(annJs):
    skudf = pd.DataFrame(annJs['__raw_Chinese_name_df'])
    return skudf
    
def evaluate(resJs, annJs, mmap=False, cn=config.debug):
    pass

def anylysis(resJs, annJs, threhold=0.8, cn=config.debug):
    pass

def visualization(resJs, annJs, threhold=0.8, cn=config.debug):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate resFile with annFile and return markdown")
    if config.debug:
        parser.add_argument(
            "--resFile",
            default="/home/dl/junk/output/mix_11/inference/coco_format_val/bbox.json",
            metavar="FILE",
            help="path to result json(support bbox and check out list)",
        )
        parser.add_argument(
            "--annFile",
            default="/home/dl/dataset/retail_product_checkout/rpc.tiny/instances_test2017.json",
            metavar="FILE",
            help="path to ground truth json(support bbox and chech out list)",
        )
        
    else:
        parser.add_argument(
            "resFile",
            metavar="FILE",
            help="path to result json(support bbox and check out list)",
        )
        parser.add_argument(
            "annFile",
            metavar="FILE",
            help="path to ground truth json(support bbox and chech out list)",
        )
    parser.add_argument(
        "--mmap",
        action='store_true',
        help="Evaluate mAP50 and mmAP",
    )
    parser.add_argument(
        "--vis",
        action='store_true',
        help="visualization after evaluate",
    )
    parser.add_argument(
        "--cn",
        action='store_true',
        help="Use raw Chinese class name, to instead of English name",
    )
    
    args = parser.parse_args()
    
    resJs = boxx.loadjson(args.resFile)
    annJs = boxx.loadjson(args.annFile)
    skudf = get_skudf(annJs)
    resTable = evaluate(resJs, annJs)
    if args.mmap:
        import pycocotools.coco as pycoco
    
    