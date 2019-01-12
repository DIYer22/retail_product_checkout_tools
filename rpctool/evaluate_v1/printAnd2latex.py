#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:25:52 2018

@author: yl
"""
r'''
	\hline
		\multirow{5}{*}{Averaged} & Baseline & 0.00\% & 	28.49	 & 1.16 & 	1.24\%	 & 0.06\%	 & 0.01\% \\
		%\cline {2-8}
		& SR w/o R (ours) & 8.19\%	 & 4.43	 & 0.36	 & 68.11\% & 	80.46\%	 & 53.08\% \\
		& SR w/o S (ours) & 9.07\%	 & 4.40	 & 0.36	 & 68.47\%	 & 80.93\%	 & 53.59\% \\
		& SR (ours) & \textbf{45.46\%} & 	\textbf{1.32}	 & \textbf{0.11}	 & \textbf{89.89\%}	 & \textbf{95.04\%}	 & \textbf{71.92\%} \\
        '''

latexTmpl = r'''
\begin{table*}[t!]
	\caption{Main results of the chekcout task on our RPC dataset.} \label{table:results}
	\centering
	\small
	%\setlength{\tabcolsep}{0.5pt}
	\begin{tabular}{c|c|c|c|c|c|c|c}
	\hline
		\textit{Difficulty mode} & \textit{Methods}           & \textit{cAcc} ($\uparrow$) & \textit{ACD} ($\downarrow$) & \textit{mCCD} ($\downarrow$) & \textit{mCIoU} ($\uparrow$) & \textit{mAP50} ($\uparrow$) & \textit{mmAP} ($\uparrow$) \\
		\hline
		\multirow{4}{*}{Easy} REPLACE_easy	\hline
		\multirow{4}{*}{Medium}  REPLACE_medium	\hline
		\multirow{4}{*}{Hard}  REPLACE_hard	\hline
		\multirow{5}{*}{Averaged} REPLACE_averaged	\hline
\end{tabular}
\end{table*}
'''
from boxx import *
from boxx import loadData, dicto, openwrite
toMethodName = dicto(
        mix='SR (ours)',
        only_gan='SR w/o S (ours)',
        no_gan='SR w/o R (ours)',
        single_fpn='Baseline'
        )

def getLatexMethodName(myName):
    for k, v in toMethodName.items():
        if myName.startswith(k):
            return v
    return myName

methodOrder = ['single_fpn', 'no_gan', 'only_gan', 'mix']
evalOrder = ['Difficulty mode', 'Methods', 'cAcc', 'ACD', 'mCCD', 'mCIoU', 'mAP50', 'mmAP']
mdOrder = ['diff', 'method', 'cAcc', 'ACD', 'mCCD', 'mCIoU', 'mAP50', 'mmAP', 'thre']
from .evaluateByBbox import diffs
diffOrder = diffs + [ 'averaged']


ROUND_NUM = 2

def toLatexCell(x, bold=False):
    if isinstance(x, float):
        s = str(round(x, ROUND_NUM))
    elif isinstance(x, str):
        if '%' in  x:
            x = x.replace('%', r'\%')
        s = x
    else:
        s = str(x)
    if bold:
        s = r'\textbf{%s}'%s
    return '&  %s  '%s
            
def exportResultMd(resTable, methodOrder=None, mdp='tmp_file_final_result.md', saveLatex=False):
    if methodOrder is None:
        methodOrder = sorted(list(resTable.values())[0].keys())
    
    mds = '## results on RPC\n'
    mds += '| %s |\n'% ' | '.join(map(str, mdOrder))
    mds += '| %s |\n'% ' | '.join(['---']*len(mdOrder))
    
    #diff = 'averaged'
    ##diff = 'easy'
    #method = 'mix'
    #method = 'no_gan'
    latex = latexTmpl
    diffStrDic = {}
    for diff in diffOrder: 
        methods = resTable[diff]
        if not methods:
            continue
        diffstr =  ''
        for method in methodOrder:
            row = methods[method]
            for eva in evalOrder:
                if eva not in  row:
                    row[eva] = '#N/A'
            
            if isinstance(row['mAP50'], float):
                row['mAP50'] = '%s%%'%round((row['mAP50']*100),2 )
                row['mmAP'] = '%s%%'%round((row['mmAP']*100),2 )
            row['Methods'] = getLatexMethodName(method, )
            bold = method == methodOrder[-1]
            if method == methodOrder[0] :
                warp = r'''%s\\
	%%\cline {2-8}
'''
            else:
                warp = '		%s\\\\\n'
            rows = warp%''.join([toLatexCell(row[eva], eva not in ['Methods']  and bold) for eva in evalOrder[1:]])
            diffstr += rows
    #        tree-row
            mds += '| %s |\n'% ' | '.join(map(str,[('**%s**' if bold else '%s')%row[eva] for eva in mdOrder]))
        diffStrDic[diff] = diffstr
        latex = latex.replace('REPLACE_%s'%diff, diffstr)
        
    s = mds
    if saveLatex:
        s += '''
## latex code
```
%s
```
'''%( latex)
    
    #openwrite(s, 'final_result.md')
    openwrite(s, mdp)
    return s

if __name__ == "__main__":
    pass



if __name__ == "__main__" :
    
    
    globKeys = ['[!f]*']
#    globKeys = ['*single*']
    #globKeys = ['oldgan_fewer*', 'mix_11_oldgan']
    
    from evaluateByBbox import junkDir
    resTableJsps = sorted(sum(map(lambda key:glob(pathjoin(junkDir, f'output/{key}/inference/coco_format_val/resTable.json')), globKeys), []))
    
#    resTables = map2(lambda p:loadjson(p).values(), resTableJsps)
    resTables = reduce(add, map2(lambda p:ll-loadjson(p).values(), resTableJsps))
#    resTables = reduce(lambda x, y:list(x.values())+list(y.values()), resTables)
    df = pd.DataFrame(resTables)
    
    def df2dicts(df, inculdeIndex=False):
        dicts = [dict(row) for index,row in df.iterrows()]
        if inculdeIndex:
            for index, dic in zip(df.index, dicts):
                dic['index'] = index
        return dicts
    resTable = defaultdict(lambda : defaultdict(lambda : {}))
    def forDf(sdf):
        d = df2dicts(sdf)[0]
        g()
        resTable[d['diff']][d['method']] = d
    ds = df.groupby(['method', 'diff']).apply(forDf)
#    resTable = loadData( 'tmp_file_resTable.pkl')
    methodOrder = sorted(set(df.method))
#    methodOrder = ['no_gan', 'only_gan', 'mix_11', ]
#    methodOrder = ['fewer-32-num6250',  'fewer-16-num12500', 'fewer-8-num25000', 'fewer-4-num50000', 'fewer-2-num100000', 'mix_11'] 
#    
#    methodOrder = ['fewer-32-num6250',  'fewer-16-num12500', 'fewer-8-num25000', 'fewer-4-num50000', 'fewer-2-num100000', 'fewer-2-num100000_60000']
#    methodOrder = ['fewer-4-num50000']+ filter2(lambda x:  'fewer-2' in x, methodOrder) + ['mix_11']
#    methodOrder = ['fewer-4-num50000']+ filter2(lambda x:  'fewer-2' in x, methodOrder) + filter2(lambda x:  'mix' in x, methodOrder) 
#    methodOrder = ['no_gan', 'only_gan', 'mix_87500', ]
#    methodOrder = ['fewer-32-num6250',  'fewer-16-num12500', 'fewer-8-num25000',"fewer-4-num50000", "fewer-2-65000", "mix_87500"]
    methodOrder = ["no_gan",]
    s = exportResultMd(resTable, methodOrder=methodOrder, saveLatex=True)
    pass
    
    
