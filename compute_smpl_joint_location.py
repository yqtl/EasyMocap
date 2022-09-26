from collections import namedtuple
import numpy as np
from easymocap.mytools.reader import read_smpl
import os
import json
# 0. load SMPL model
from easymocap.smplmodel import load_model
from os.path import join
import argparse
from easymocap.config import Config, load_object
parser = parser = argparse.ArgumentParser('EasyMocap commond line tools')
parser.add_argument('--nf', type=int, default=None)
args = parser.parse_args()
# 1. load parameters
nf=args.nf
step=1
smpl_input_path='smpl/smpl'
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
def read_smpl(nf):
        outname = join(smpl_input_path, '{:06d}.json'.format(nf))
        assert os.path.exists(outname), outname
        datas = read_json(outname)
        outputs = []
        for data in datas:
            for key in ['Rh', 'Th', 'poses', 'shapes']:
                data[key] = np.array(data[key])
            outputs.append(data)
        return outputs
infos = read_smpl(nf*step)
#print(infos)
'''
[{'id': 0, 'Rh': array([[ 1.279, -1.263, -1.197]]), 'Th': array([[0.42 , 0.285, 1.168]]), 'poses': array([[ 0.   ,  0.   ,  0.   , -0.067,  0.093,  0.162, -0.147, -0.082,
        -0.096,  0.   ,  0.   , -0.   ,  0.022,  0.044, -0.073,  0.094,
        -0.049,  0.042, -0.   ,  0.   ,  0.   , -0.043,  0.136, -0.004,
        -0.004, -0.144,  0.024,  0.034,  0.   ,  0.   , -0.   ,  0.   ,
        -0.   ,  0.   ,  0.   , -0.   , -0.288,  0.39 ,  0.043,  0.   ,
         0.   ,  0.   , -0.   ,  0.   ,  0.001,  0.281,  0.353,  0.019,
        -0.03 , -0.1  , -0.368, -0.027,  0.071,  0.426, -0.018, -0.158,
         0.065, -0.01 ,  0.091,  0.013, -0.   ,  0.   ,  0.   ,  0.   ,
        -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ,  0.   ]]), 'shapes': array([[ 0.188, -0.178,  0.054,  0.178,  0.027,  0.035, -0.021, -0.022,
         0.018, -0.015]])}]
'''
for info in infos:
    #config = Config.load(args.cfg)
    #the following line has no effect??
    body_model=load_model('male','smpl')
    #body_model = load_object(config.module, config.args)
# 2. compute joints
    #params = body_model.init_params(1)
    pose=info["poses"]
    Rh=info["Rh"]
    Th=info["Th"]
    shapes=info["shapes"]
    #global_orient = pose[:, :3]
#taking the global orientation of the body (the first 3 columns of the pose matrix)
    #body_pose = pose[:, 3:]
    body_pose=pose
#taking the body pose (from the 4th columns to the end of the pose matrix)
    body_params = {
    'poses':body_pose,
    'shapes': shapes,
    'Rh':Rh,
    'Th':Th
    #'transl': trans
    }
    #smpl_model.forward(**body_params)

#for info in infos:
    #get parameter 
    #print(data)
    
#use_joints is not a member of body_model, it is a member of SMPLLAYER in body_param.load_model(), which does not define "use_joints" argument
    #joints = body_model(return_verts=True, return_tensor=False, **body_params)[0]
    joints_location=body_model(return_verts=False, return_tensor=False, **body_params)[0]
    #print(joints)
    print(joints_location)
# 3. compute vertices
#vertices = body_model(return_verts=True, return_tensor=False, **infos)[0]