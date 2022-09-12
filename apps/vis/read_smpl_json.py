#import pyrender # first import the pyrender
from collections import namedtuple
import numpy as np
from easymocap.mytools.reader import read_smpl
import os
import json
from easymocap.dataset.base import MVBase
from easymocap.dataset.config import CONFIG
# 0. load SMPL model
from easymocap.smplmodel import load_model
from os.path import join
from easymocap.mytools.cmd_loader import load_parser
from easymocap.config import Config, load_object

#parser = load_parser()
#parser.add_argument('--gender', type=str, default='neutral', choices=['male', 'female', 'neutral'])
#parser.add_argument('--model', type=str, default='smpl', choices=['none', 'smpl', 'smplx'])
#args = parser.parse_args()

#body_model = load_model(args.gender, model_type=args.model)
# 1. load parameters
nf=0
step=0
smpl_output_path='smpl/smpl'
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
def read_smpl(nf):
        outname = join(smpl_output_path, '{:06d}.json'.format(nf))
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
#config = Config.load(args.cfg)
body_model=load_model('male','smpl')
#body_model = load_object(config.module, config.args)
# 2. compute joints
params = body_model.init_params(1)
#for data in infos:
    #get parameter 
    #print(data)
    
'''
{'poses': array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]]), 'shapes': array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'Rh': array([[0., 0., 0.]]), 'Th': array([[0., 0., 0.]])}
'''

joints = body_model(return_verts=False, return_tensor=False, *infos)[0]
print(joints)
# 3. compute vertices
vertices = body_model(return_verts=True, return_tensor=False, **infos)[0]

    
    
Person = namedtuple('Person', ['vertices', 'keypoints3d'])

def inBound(keypoints3d, bound):
    if bound is None:
        return True
    valid = np.where(keypoints3d[:, -1] > 0.01)[0]
    kpts = keypoints3d[valid]
    crit = (kpts[:, 0] > bound[0][0]) & (kpts[:, 0] < bound[1][0]) &\
        (kpts[:, 1] > bound[0][1]) & (kpts[:, 1] < bound[1][1]) &\
        (kpts[:, 2] > bound[0][2]) & (kpts[:, 2] < bound[1][2])
    if crit.sum()/crit.shape[0] < 0.8:
        return False
    else:
        return True 

def visualize(path, sub, out, mode, rend_type, args):
    config = CONFIG[mode]
    no_img = False
    dataset = MVBase(path, cams=sub, config=config,
        undis=args.undis, no_img=no_img, out=out)
    dataset.skel_path = args.skel
    if rend_type in ['skel']:
        from visualize.skelmodel import SkelModel
        body_model = SkelModel(config['nJoints'], config['kintree'])
    elif rend_type in ['mesh']:
        from smplmodel import load_model
        body_model = load_model(args.gender, model_type=args.model)
        smpl_model = body_model
    elif rend_type == 'smplskel':
        from smplmodel import load_model
        smpl_model = load_model(args.gender, model_type=args.model)
        from visualize.skelmodel import SkelModel
        body_model = SkelModel(config['nJoints'], config['kintree'])
    
    dataset.writer.save_origin = args.save_origin
    start, end = args.start, min(args.end, len(dataset))
    bound = None
    if args.scene == 'none':
        ground = create_ground(step=0.5)
    elif args.scene == 'hw':
        ground = create_ground(step=1, xrange=14, yrange=10, two_sides=False)
        bound = [[0, 0, 0], [14, 10, 2.5]]
    else:
        ground = create_ground(step=1, xrange=28, yrange=15, two_sides=False)
    for nf in tqdm(range(start, end), desc='rendering'):
        images, annots = dataset[nf]
        if rend_type == 'skel':
            infos = dataset.read_skel(nf)
        else:
            infos = dataset.read_smpl(nf)
        # body_model: input: keypoints3d/smpl params, output: vertices, (colors)
        # The element of peopleDict must have `id`, `vertices`
        peopleDict = {}
        for info in infos:
            if rend_type == 'skel':
                joints = info['keypoints3d']
            else:
                joints = smpl_model(return_verts=False, return_tensor=False, **info)[0]
            if not inBound(joints, bound):
                continue
            if rend_type == 'smplskel':
                joints = smpl_model(return_verts=False, return_tensor=False, **info)[0]
                joints = np.hstack([joints, np.ones((joints.shape[0], 1))])
                info_new = {'id': info['id'], 'keypoints3d': joints}
                vertices = body_model(return_verts=True, return_tensor=False, **info_new)[0]
            else:
                vertices = body_model(return_verts=True, return_tensor=False, **info)[0]
            peopleDict[info['id']] = Person(vertices=vertices, keypoints3d=None)
        dataset.vis_smpl(peopleDict, faces=body_model.faces, images=images, nf=nf, 
            sub_vis=args.sub_vis, mode=rend_type, extra_data=[ground], add_back=args.add_back)

if __name__ == "__main__":
    from mytools.cmd_loader import load_parser
    parser = load_parser()
    parser.add_argument('--type', type=str, default='mesh', choices=['skel', 'mesh', 'smplskel'])
    parser.add_argument('--scene', type=str, default='none', choices=['none', 'zjub', 'hw'])
    parser.add_argument('--skel', type=str, default=None)
    parser.add_argument('--add_back', action='store_true')
    parser.add_argument('--save_origin', action='store_true')
    args = parser.parse_args()
    visualize(args.path, args.sub, args.out, args.body, args.type, args)