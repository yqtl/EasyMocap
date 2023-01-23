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
from glob import glob
def load_single_file_motions(filename):
    single_file_motions = {}    
    infos = read_smpl(filename)
    for data in infos:
        pid = data['id']
        if pid not in single_file_motions.keys():
            single_file_motions[pid] = []
        single_file_motions[pid].append(data)
    keys=list(single_file_motions.keys())
    for pid in single_file_motions.keys():
        single_file_motions[pid] = merge_params(single_file_motions[pid])
    return single_file_motions
def compute_joint_locations(body_params):
    joints_location=body_model(return_verts=False, return_tensor=False, **body_params)[0]
    return joints_location
def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output
def write_keypoints3d(data, filename):
    newfilename = filename.replace('.json', '_keypoints3d.json')
    with open(newfilename, 'w') as f:
        json.dump(data, f, indent=4)


path='/mimer/NOBACKUP/groups/snic2022-22-770/Boxing2/seq1/output-track/smpl_neutral'
filenames = sorted(glob(join(path, '*.json')))
body_model=load_model('neutral','smpl')
for filename in filenames:
    #print(filename)
    single_file_motions=load_single_file_motions(filename)
    write_data=[]
    for pid in single_file_motions.keys():
        body_params = {'poses':single_file_motions[pid]['poses'].reshape(1,72),'shapes': single_file_motions[pid]['shapes'].reshape(1,10),'Rh':single_file_motions[pid]['Rh'].reshape(1,3),'Th':single_file_motions[pid]['Th'].reshape(1,3)}
        joints=compute_joint_locations(body_params)
        #joints are numpy array of shape (25, 3)
        #construct a dictionary with keys 'id', 'type', 'keypoints3d'
        #append dictionary to the list write_data
        data={}
        data['id']=pid
        data['type']='body25'
        data['keypoints3d']=joints.tolist() #convert numpy array to list
        write_data.append(data)
        #write_data = [{'id': pid, 'type':'body25','keypoints3d': joints.tolist()}]
        if 'keypoints3d' not in single_file_motions[pid].keys():
            single_file_motions[pid]['keypoints3d'] = []
        #split keypoints3d to 25 items in a list 
        single_file_motions[pid]['keypoints3d'].append(joints)
        
        #single_file_motions[pid]['keypoints3d']=(joints)
        #construct a new json file
        single_file_motions[pid]['id']=pid
        single_file_motions[pid]['type']='body25'
    write_keypoints3d(write_data, filename)
