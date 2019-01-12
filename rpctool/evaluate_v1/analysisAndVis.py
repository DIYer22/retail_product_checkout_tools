#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:53:41 2018

@author: yl
"""
#from evaluateByBbox import *
#from evaluateByBbox import resJsps
#from config import skuDf
from boxx import *
from boxx import loadjson, pd, Counter, p
from boxx import reduce, add, ll, tree, x_, os, pathjoin

skuDf = pd.read_csv(os.path.abspath(pathjoin(__file__, '../../sku_info_generated.csv')))

#resJsp = resJsps[0]
#resJsp = r'c:/D/junk\output\valAsTrain_raw\bbox_coco_2017_val_results.json'
#resJsp = r'c:/D/junk\output\testAsTrain_raw\bbox_coco_2017_val_results.json'
#resJsp = r'c:/D/junk\output\mix_11_oldgan\bbox_coco_2017_val_results.json'
resJsp = "/home/dl/junk/output/mix_11/inference/coco_format_val/bbox.json"
resJsp = "/home/dl/junk/output/testAsTrain/inference/coco_format_val/bbox.json"
print(resJsp)
thre = .784321 # mix 11
thre = 0.84 # testAsTrain
diff = 'easy'
#diff = 'medium'
#diff = 'hard'
diff = 'all'

getWrong = True
#getWrong = False

xuanMai = [i + 1 for i in [135,136,137]]

if   1:
    valJsp = '../checkout_submission_tools/tmp_file_json/instances_test2017.json'
    coco = loadjson(valJsp)
    
    
    imgds = coco['images']
    imgdf = pd.DataFrame(imgds)
    imgdf = imgdf.set_index('id')
    imgdf['id'] = imgdf.index
    
    if diff!='all':
        imgdf = imgdf[imgdf['level'] == diff]
    imgIds = set(imgdf.id)
    
    gtanns = coco['annotations']
    gtdf = pd.DataFrame(gtanns)
    gtdf = gtdf[gtdf.image_id.isin(imgIds)]
#    gtdf['level'] = gtdf.image_id.apply(lambda idd: imgdf.loc[idd]['level'])
    gtdf['fname'] = gtdf.image_id.apply(lambda idd: imgdf.loc[idd]['file_name'])
    
    rebboxs = loadjson(resJsp)
    redf = pd.DataFrame(rebboxs)
    redf = redf[redf.score>thre]
    redf = redf[redf.image_id.isin(imgIds)]
    redf['fname'] = redf.image_id.apply(lambda idd: imgdf.loc[idd]['file_name'])
    
    
    # TODO
    reImgIds = set(redf.image_id)
    if imgIds - reImgIds:
        print("less in bbox.json")
        tree(imgIds - reImgIds)
        gtdf = gtdf[gtdf.image_id.isin(reImgIds)]
        

def getCnNameByCatId(catId, ):
    return skuDf.loc[catId-1]['name']
def getCounter(name, all=False):
    imgId = imgdf[imgdf.file_name.apply(lambda x:name in x )].iloc[0].id
    gt, re = Counter(gtdf[gtdf.image_id==imgId].category_id.apply(getCnNameByCatId)), Counter(redf[redf.image_id==imgId].category_id.apply(getCnNameByCatId))
    if all:
        return gt, re
    return gt-re, re-gt
    
gtct = gtdf.groupby('fname').apply(lambda sdf: Counter(sdf.category_id))
rect = redf.groupby('fname').apply(lambda sdf: Counter(sdf.category_id))

topk = 100

if getWrong:
    wrongsetdf = (rect-gtct)+(gtct-rect)
    wrongdf = wrongsetdf.apply(lambda ct:sum(ct.values()))
    wrongdf = wrongdf[wrongdf.apply(bool)]
    wrongdf = wrongdf.sort_values(ascending=False)
    
    #wrongsetdf[wrongsetdf.apply(lambda x:bool(len(set(x).intersection(xuanMai))))]
    
    
    wrongdf = wrongdf.iloc[:topk]
    
    cmd = '\n'.join([f"cp all_croped/{n} analysis/{diff}Wrong/{w}_{n} " for n,w in wrongdf.items()])
    rscp = '\n'.join([f"scp bpp:/home/yanglei/dataset/checkout-data/check_image/all_croped/{n} {w}_{n} " for n,w in wrongdf.items()])
else:
    rightdf = rect.apply(lambda ct:sum(ct.values()))
    rightdf = rightdf[gtct == rect]
    rightdf = rightdf.sort_values(ascending=False)
    rightdf = rightdf.iloc[:topk]
    cmd = '\n'.join([f"cp all_croped/{n} analysis/goodRight/{diff}_{w}_{n} " for n,w in rightdf.items()])

#p-cmd
#print(cmd)
if __name__ == "__main__":
    
    clasCd = reduce(add, wrongsetdf)
    
    badClas = sorted(clasCd.items(), key=x_[1])
    badClas = [(skuDf.loc[l[0]-1]['name'], l[1]) for l in badClas]
    tree - badClas
    pass
if __name__ == "__main__":
    # save gt_bbox_clas_distri.xlsx
    # pd.DataFrame( sorted((Counter(gtdf.category_id)+Counter(gtdf2.category_id)).items(), key=lambda x:-x[1])).to_excel('tmp_file_gt_bbox_clas_distri.xlsx')
    
    
    clasMissCd = reduce(add, gtct-rect) # FN
    clasFpCd = reduce(add, rect-gtct)
    
    clasXls = pd.DataFrame(gtdf.groupby('category_id')['id'].count())
    clasXls['num'] = clasXls['id']
    clasXls.pop('id')
    
    clasXls['sku_class'] = clasXls.index.map(lambda x: skuDf.loc[x-1]['sku_class'])
    clasXls['sku_name'] = clasXls.index.map(lambda x: skuDf.loc[x-1]['sku_name'])
    def getCdDf(clasCd, tag):
        dic = dict(clasCd)
        for k in clasXls.index:
            if k not in dic:
                dic[k] = 0
        
        df = pd.DataFrame(ll-(dic.items()))
        df[['catId','cd']] = df[:]
#        df['cn'] = df.catId.apply(lambda idd:skuDf.loc[idd-1]['name'])
#        df['en'] = df.catId.apply(lambda idd:skuDf.loc[idd-1]['sku_name'])
#        df['clas'] = df.catId.apply(lambda idd:skuDf.loc[idd-1]['sku_class'])
        df = df.set_index('catId')
        clasXls[tag] = df['cd']/clasXls.num
        clasXls[tag+'_num'] = df['cd']
        return df
    getCdDf(clasFpCd, 'fp')
    getCdDf(clasMissCd, 'miss')
    clasXls['summ'] = clasXls.fp + clasXls.miss
    clasXls = clasXls.sort_values('summ', ascending=False)
    
    import matplotlib.pyplot as plt
#    plt.show([1,2])
    clasXls.summ.plot.bar()
    plt.show()
    
    badClas = sorted(clasCd.items(), key=lambda x:-x[1], )
    badClas = [(skuDf.loc[l[0]-1]['sku_name'], l[1]) for l in badClas]
    tree - badClas
    pass

    cid2ch = dict(zip(skuDf.category_id, skuDf['name']))
    clasXls['ch'] = clasXls.index.map(lambda x:cid2ch[x])

if 1:
    pass
    K = 200
    imgCtDf = pd.DataFrame()
    imgCtDf['gtct'] = gtct
    imgCtDf['rect'] = rect
    
    counter2list = lambda ct: [ct.get(i, 0) for i in range(1, K+1)]
    imgCtDf['array'] = imgCtDf.apply(lambda d:np.array([counter2list(d.rect), counter2list(d.gtct),]) , 1)    
    imgCtDf['maxx'] = imgCtDf['array'].apply(lambda arr: arr.max(0))  
    imgCtDf['minn'] = imgCtDf['array'].apply(lambda arr: arr.min(0))
    mciouMatrix = imgCtDf.minn.sum()/imgCtDf.maxx.sum()
    mciou = dict(enumerate(mciouMatrix, 1))
    clasXls['ciou'] = clasXls.index.map(lambda idd:mciou[idd])
    
    diffDf = (rect-gtct)+(gtct-rect)
    acdDf = diffDf.apply(lambda ct:sum(ct.values()))
    
    mccdDf = pd.Series(diffDf.sum())/pd.Series(gtct.sum())
    mccdDf = mccdDf.fillna(0)
    gtNumDf = gtct.apply(lambda ct:sum(ct.values()))
    evalDic = dicto(
            cAcc=(gtct == rect).mean(), 
            mCIoU = mciouMatrix.mean(), 
            ACD=acdDf.mean(), 
            mCCD=mccdDf.mean(),
            )
    tree-evalDic
    plot(mciouMatrix,1)
    
    
    
    superClasXls = clasXls.groupby('sku_class').sum()
    superClasXls['fp'] = superClasXls.fp_num/superClasXls.num
    superClasXls['miss'] = superClasXls.miss_num/superClasXls.num
    superClasXls['summ'] = superClasXls.fp + superClasXls.miss
    superClasXls = superClasXls.sort_values('summ', ascending=False)
    
#    with pd.ExcelWriter('tmp_file.xlsx',engine='openpyxl') as f:
#        clasXls[['miss','fp']].to_excel(f, sheet_name='clasXls')
#        superClasXls[['miss','fp']].to_excel(f, sheet_name='superClasXls')
    fpdf = clasXls.sort_values("fp_num", ascending=False).fp_num
    missdf = clasXls.sort_values("miss", ascending=False).miss
    cioudf = clasXls.sort_values("ciou", ascending=False).ciou
    
    
    supfpdf = superClasXls.sort_values("fp_num", ascending=False).fp_num
    supmissdf = superClasXls.sort_values("miss", ascending=False).miss
    supcioudf = superClasXls.sort_values("ciou", ascending=False).ciou/clasXls.groupby('sku_class').num.count()
    
    with pd.ExcelWriter('tmp_file_rpc_pic.xlsx',engine='openpyxl') as f:
        fpdf.to_excel(f, sheet_name='fpClas')
        missdf.to_excel(f, sheet_name='missClas')
        cioudf.to_excel(f, sheet_name='ciouClas')
        
        supfpdf.to_excel(f, sheet_name='fpSuperClas')
        supmissdf.to_excel(f, sheet_name='missSuperClas')
        supcioudf.to_excel(f, sheet_name='ciouSuperClas')
    
    