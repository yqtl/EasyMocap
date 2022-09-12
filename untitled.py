# 0. load SMPL model
import easymocap
from easymocap.smplmodel import load_model
import easymocap.dataset
from easymocap.dataset.base import read_smpl
load_model
load_model(gender='male', use_cuda=True, model_type='smpl', skel_type='body25', device=None, model_path='data/smpl/')

# 1. load parameters
#read_smpl('smpl/smpl/')
infos = read_smpl('smpl/smpl/000000.json')
# 2. compute joints
joints = body_model(return_verts=False, return_tensor=False, **info)[0]
# 3. compute vertices
vertices = body_model(return_verts=True, return_tensor=False, **info)[0]