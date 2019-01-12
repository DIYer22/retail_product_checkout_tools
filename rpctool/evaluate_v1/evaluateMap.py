# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:05:56 2018

@author: yl
"""


    
    
from boxx import *
import matplotlib.pyplot as plt
from pycocotools.coco import COCO as rawCOCO
from pycocotools.cocoeval import  COCOeval as rawCOCOeval

import  pycocotools.coco as pycoco
pycoco.print = lambda *l,**kv:0
import pycocotools.cocoeval as pycocoeval
pycocoeval.print = lambda *l,**kv:0
from pycocotools.coco import defaultdict, time, json, maskUtils, copy
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)



class COCO(rawCOCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            if isinstance(annotation_file, dict):
                dataset = annotation_file
            else:
                dataset = json.load(open(annotation_file, 'r'))
            
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
            
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        imgIds = self.getImgIds()
        anns = [bb for bb in anns if bb['image_id'] in imgIds]
        annsImgIds = [ann['image_id'] for ann in anns]
        
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

class COCOeval(rawCOCOeval):
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
#            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
#            g()
#            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
#            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
#            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
#            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
#            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
#            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
#            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
#            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
#            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
#            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        self.stats = summarize()
        return dict(zip(['mmAP', 'mAP50'], self.stats[:2] ))

def evalMap(annFile, resFile, toStr=False, annType='bbox'):
    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes(resFile)
    
    
    
#    imgIds=sorted(cocoGt.getImgIds())
#    imgIds=imgIds[0:100]
#    imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
#    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    res = cocoEval.summarize()
    if toStr:
        return {k:'%s%%'%round(v*100,2) for k,v in res.items()}
    return res
if __name__ == "__main__":
    from evaluateByBbox import *
    
    
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
#    print( 'Running demo for *%s* results.'%(annType))
    
    
    
    #initialize COCO ground truth api
    dataDir='../'
    dataType='val2014'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    
    #initialize COCO detections api
    resFile='%s/results/%s_%s_fake%s100_results.json'
    resFile = resFile%(dataDir, prefix, dataType, annType)
    
    annFile = cocoCheckBboxAnnJsp
    resFile = resJsp
    res = evalMap(annFile, resFile)
    tree-res
    
    
    