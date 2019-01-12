#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Tools for Retail-Product-Checkout-Dataset(RPC)
'''
__version__ = "0.1.1"
__short_description__ = "Tools for Retail-Product-Checkout-Dataset(RPC)"
__author__ = "DIYer22, Cui Quan, "
__author_email__ = "ylxx@live.com"
__github_url__ = "https://github.com/DIYer22/retail_product_checkout_tools"
__support__ = "https://github.com/DIYer22/retail_product_checkout_tools/issues"

import pandas as pd
from .rpc_config import config
#config.debug = True

def get_skudf(annJs):
    skudf = pd.DataFrame(annJs['__raw_Chinese_name_df'])
    return skudf

def get_catdf(annJs, cn=config.debug):
    catdf = pd.DataFrame(annJs['categories'])
    if cn:
        skudf = get_skudf(annJs)
        catdf = pd.DataFrame()
        catdf[['name', 'id', 'supercategory']] = skudf[['name', 'category_id', 'clas']]
        
    return catdf
    
#def evaluate(resJs, annJs, mmap=False, cn=config.debug):
#    pass

def evaluate_v1_interface(resJs, annJs, mmap=False, cn=config.debug, method="default", log=True):
    from .evaluate_v1 import evaluateByBbox as v1
    v1.getMmap = mmap
    from boxx import pathjoin, tmpboxx, savejson
    resJsp = pathjoin(tmpboxx(), "rpc_res.json")
    annJsp = pathjoin(tmpboxx(), "rpc_ann.json")
    savejson(resJs, resJsp)
    savejson(annJs, annJsp)
    resTable = v1.evaluateByJsp(resJsp, annJsp, method=method, log=log)
    df = pd.DataFrame([resTable[diff] for diff in ['easy', 'medium', 'hard', 'averaged']])
    
    column_names = [col for col in config.column_names if col in df]
    from boxx import Markdown
    md = Markdown(df[column_names])
    return md

evaluate = evaluate_v1_interface
def anylysis(resJs, annJs, threhold=0.8, cn=config.debug):
    pass

def visualization(resJs, annJs, threhold=0.8, cn=config.debug):
    pass


if __name__ == '__main__':
    pass