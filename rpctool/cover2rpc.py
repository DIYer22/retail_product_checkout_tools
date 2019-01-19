#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Tools for cover not standard coco format json data(source) to standard Retail-Product-Checkout-Dataset format(target)
'''
import boxx

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Cover not standard coco format json data(source) to standard Retail-Product-Checkout-Dataset format(target)")

parser.add_argument(
    "sourceFile",
    metavar="FILE",
    help="path to not standard coco format json",
)
parser.add_argument(
    "targetFile",
    metavar="FILE",
    help="path to RPC ground truth json",
)


def cover2rpc(sourceFile, targetFile):
    
    from boxx import openwrite, openread, df2dicts, zip2
    sjs = boxx.loadjson(sourceFile)
    tjs = boxx.loadjson(targetFile)
    
    openwrite(openread(sourceFile), sourceFile+'.before.cover2rpc.bak')
    
    imgdf = pd.DataFrame(sjs["images"])
    anndf = pd.DataFrame(sjs["annotations"])
    
    imgid2len = anndf.groupby('image_id').apply(lambda sdf:[len(sdf), len(set(sdf.category_id)), ])
    
    imgdf['instance_num'] = imgdf.id.apply(lambda i:imgid2len.loc[i][0])
    imgdf['class_num'] = imgdf.id.apply(lambda i:imgid2len.loc[i][1])
    imgdf = imgdf.sort_values(['instance_num'])
    n = len(imgdf)
    
    indexDic = dict(zip2(range(n//3), ['easy']*n)+zip2(range(n//3, 2*n//3), ['medium']*n)+zip2(range(2*n//3, n), ['hard']*n))
    imgdf['_ind'] = range(n)
    imgdf['level'] = imgdf._ind.apply(lambda i:indexDic[i])
    imgdf.pop('_ind')
#    imgdf.iloc[:n//3].level = 'easy'
#    imgdf.iloc[n//3:n*2//3].level = 'medium'
#    imgdf.iloc[n*2//3:].level = 'hard'
    assert all(imgdf.iloc[n*2//3:].level == 'hard'), imgdf.iloc[-1]
    
    imgdf.pop('instance_num')
    imgdf.pop('class_num')
    dicts = df2dicts(imgdf)
    tjs['images'] = dicts
    tjs['annotations'] = sjs["annotations"]
    return boxx.savejson(tjs, sourceFile)
if __name__ == '__main__':
#    cover2rpc("/home/dl/junk/val.json", "/home/dl/junk/instances_val2017.json")
    args = parser.parse_args()
    cover2rpc(args.sourceFile, args.targetFile)