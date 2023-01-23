#read all smpl and compute joints, save in a large matrix
from collections import namedtuple
import numpy as np
import glob
from easymocap.mytools.reader import read_smpl
import os
import json
from easymocap.smplmodel import load_model
from os.path import join
import argparse
from easymocap.config import Config, load_object

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
def read_smpl(filename):        
    assert os.path.exists(filename), 'File not found: {}'.format(filename)
    datas = read_json(filename)
    if isinstance(datas, dict):
        datas = datas['annots']
    outputs = []
    for data in datas:
        #for key in ['id', 'Rh', 'Th', 'poses', 'shapes', 'expression', 'handl', 'handr']:
        for key in ['Rh', 'Th', 'poses', 'shapes']:
            if key in data.keys():
                data[key] = np.array(data[key])
        outputs.append(data)
    return outputs
def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output
def load_motions(path):
    from glob import glob
    filenames = sorted(glob(join(path, '*.json')))
    #print(filenames)
    motions = {}
    # for filename in filenames[300:900]:
    for filename in filenames:
        infos = read_smpl(filename)
        for data in infos:
            pid = data['id']
            if pid not in motions.keys():
                motions[pid] = []
            motions[pid].append(data)
    keys = list(motions.keys())
    for pid in motions.keys():
        motions[pid] = merge_params(motions[pid])
    return motions
def load_smpl_params(datapath):
    motions = load_motions(datapath)
    return motions
def compute_joint_locations(body_params):
    joints_location=body_model(return_verts=False, return_tensor=False, **body_params)[0]
    return joints_location

motions = load_smpl_params('output-track/smpl')
body_model = load_model('neutral','smpl')


for pid in motions.keys():
    #body_params = {'poses':motions[pid]["poses"],'shapes': shapes,'Rh':Rh,'Th':Th}
    #motions[pid]
    #joints_all=np.array(joints)
    for i in range(0,len(motions[pid]['poses'])):
        body_params2 = {'poses':motions[pid]['poses'][i].reshape(1,72),'shapes': motions[pid]['shapes'][0].reshape(1,10),'Rh':motions[pid]['Rh'][i].reshape(1,3),'Th':motions[pid]['Th'][i].reshape(1,3)}
        joints=compute_joint_locations(body_params2)
        if i ==0:
            joints_all=np.array(joints)
        else:
            joints_all= np.vstack((joints_all,joints))
        #output[key] = np.vstack([v[key] for v in param_list])
    if 'keypoints3d' not in motions[pid].keys():
        motions[pid]['keypoints3d'] = []
    motions[pid]['keypoints3d']=(joints_all)
        #motions[pid]['keypoints3d']=np.vstack([joints])
