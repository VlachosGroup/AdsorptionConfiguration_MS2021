# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:35:37 2019

@author: user
"""
import json
from util import Surface
from glob import glob
from tqdm import tqdm
from datetime import datetime
from time import time
import pickle



surfacepath = './Surfaces/'
surfs = {p[11:-8]:Surface(p,name=p[11:-8]) for p in glob(surfacepath+'*.CONTCAR')}
def func(inputs):
    sp, s = inputs
    #surf = Surface(sp,name=sp[17:-8])
    surf = surfs[sp[11:-8]]
    data = []
    for proj in surf.GetProjection(s):
        try:
            data.append((proj[0],surf.name,proj[1]))
        except:
            continue
    return data

mapper = map

outpath = './Output/Database.pkl'
surf_paths = glob(surfacepath+'/*.CONTCAR')
input_jsons ='Output/Size1.json'


n=0
inputss = []
smiles = json.load(open(input_jsons))
for s in tqdm(smiles):
    if '*' not in s: continue
    for sp in surf_paths:
        inputss.append((sp,s))

gen = mapper(func, inputss)
ndata = len(inputss)
print(datetime.now().strftime("[%H:%M:%S]"),'0','/',ndata)
data = []; t = time()
for i,r in enumerate(gen):
    data += r
    if i != 0 and i %1000 == 0:
        print(datetime.now().strftime("[%H:%M:%S]"),i,'/',ndata,\
        '| %.2f/1000 sec/cif |'%((time()-t)/i*1000), '~%.2f sec left'%((ndata-i)/i*(time()-t)))


pickle.dump(data,open(outpath,'wb'))
