# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:11:47 2018

@author: yl
"""

from .evaluate import evaluate
from boxx import *
from boxx import cloud, addPathToSys, loadjson, reduce, add, findints, winYl, g

    
from boxx import dirname, np, pd, basename, map2, p, glob, ignoreWarning, x_, tree, defaultdict
from boxx import saveData, savejson
from copy import deepcopy

ignoreWarning()
K = 200
debug = False
#debug = True
getMmap = True
#getMmap = False
printOrder = ['cAcc', 'ACD', 'mCIoU', 'mCCD']

gtJsp = '../checkout_submission_tools/tmp_file_json/instances_test2017.json'
#gtJsp = '../checkout_submission_tools/tmp_file_json/instances_val2017.json'
#gtJsp = '/home/dl/dataset/retail_product_checkout/new_sku_data/annotations/instances_val2017.json'


globKeys = ['valAsTrain_*']
globKeys = ['mix_*']
globKeys = ['[!f]*']
globKeys = ['few*']
globKeys = ['fewer-2-65000','fewer-2-67500',]
globKeys = ['mix_850000', 'mix_87500']
globKeys = ['single1_on_check_??', ]
globKeys = ['*single*', ]
globKeys = ["single1_on_check_10000",]
globKeys = ["valAndNoGan",]
globKeys = ["testA*"]
globKeys = ["mix_11"]
globKeys = ["valRotateCocopre"]

#[imgd.update({'count':list([0]*K)}) for imgd in gtJs['images']]

def getGtCounts(gtJs):
    imgKv = {imgd['id']:imgd for imgd in gtJs['images']}
    for bbox in gtJs['annotations']:
        imgd = imgKv[bbox['image_id']]
        imgd['count'] = imgd.get('count', []) + [bbox['category_id']-1]
    gt_counts = {imgd['file_name']:imgd.get('count', []) for imgd in imgKv.values()}
    return gt_counts, imgKv

def evaluateByJson(resJs, gtJs, log=False):
    gtJs = deepcopy(gtJs)
    gt_counts, imgKv = getGtCounts(gtJs)
    resJs = [d for d in resJs if d['image_id'] in imgKv]
    def evaluateByThrehold(threhold):
        [d['countRes'].clear() for d in imgKv.values() if 'countRes' in d]
        for resd in resJs:
            if resd['score'] < threhold:
                continue
            imgd = imgKv[resd['image_id']]
            imgd['countRes'] = imgd.get('countRes', []) + [resd['category_id']-1]
        
        pred_counts = {imgd['file_name']:imgd.get('countRes', []) for imgd in imgKv.values()}
        scores = evaluate(pred_counts, gt_counts, log=False)
        scores['thre'] = threhold
        return scores
    
    thres = (.01,.99)
    for i in range(debug or 4):
        xs = np.linspace(min(thres),max(thres), [10, 4][debug])
        df = pd.DataFrame(map2(evaluateByThrehold, xs, ))
        thres = df.loc[df.cAcc.nlargest(3).index].thre
        
    row = df.loc[df.cAcc.argmax()]
    p-"row.thre = %s"% row.thre
#    p-row
    row[printOrder] = [round(row[k], 4) for k in printOrder]
    
    row['cAcc'] = '%s%%'%round((row['cAcc']*100),2 )
    row['mCIoU'] = '%s%%'%round((row['mCIoU']*100),2 )
    
    row = {k:round(v,2) if isinstance(v, float) else v for k,v in row.items()}
    row = dict(row)
    if getMmap:
        from .evaluateMap import evalMap
        mapd = evalMap(gtJs, resJs, toStr=True)
        row.update(mapd)
    return row

def evaluateByJsp(resJsp, gtJsp, log=True, method=None, ):
    if method is None:
        method = basename(dirname(dirname(dirname(resJsp))))
#        method=basename(dirname(resJsp))
    resTable = defaultdict(lambda :{})
    
    
    resJs = loadjson(resJsp)
    gtJs = loadjson(gtJsp)
    
    diff='averaged'
    row = evaluateByJson(resJs, gtJs, )
    
    row['method'] = method
    row['diff'] = diff
    resTable[diff] = dict(row)
    tree-row
    for diff in diffs:
        coco = loadjson(gtJsp)
        coco['images'] = [d for d in coco['images'] if d['level']==diff]
        imgIds = [d['id'] for d in coco['images']]
        coco['annotations'] = [bb for bb in coco['annotations'] if bb['image_id'] in imgIds]
        resJs = loadjson(resJsp)
        row = evaluateByJson(resJs, coco,)
        
        row['method'] = method
        row['diff'] = diff
        resTable[diff] = dict(row)
        tree-row
        
    resdir = dirname(resJsp)
    savejson(resTable, pathjoin(resdir, 'resTable.json'))
    return resTable

diffs = ['easy', 'medium', 'hard']


#gtJsp = cocoCheckBboxAnnJsp 


junkDir = '/home/dl/junk'
if winYl:
    junkDir = 'c:/D/junk'


resJsps = sorted(sum(map(lambda key:glob(pathjoin(junkDir, f'output/{key}/inference/coco_format_val/bbox.json')), globKeys), []))
if __name__ == "__main__":

    from  printAnd2latex import exportResultMd
    
    for resJsp in resJsps:
        resTable = evaluateByJsp(resJsp, gtJsp)
        
        resOld =  {k:{'mix':v} for k,v in resTable.items()}
        exportResultMd(resOld)
        
#import matplotlib.pyplot as plt
#for s in scoress.T:
#    plt.plot(xs, s)
#    plt.show()   loga(ll-imgdf[imgdf.wh.apply(x_[0])<1750].wh)
'''no_gan threhold=0.6473684210526316
Score1 is 0.1725, Score2 is 2.7268, Score3 is 0.3765, Score4 is 0.3973'''
    
