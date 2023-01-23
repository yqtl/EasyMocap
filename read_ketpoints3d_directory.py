#read multi-cam reconstructed keypoints3d
from glob import glob
from os.path import join
def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        ret = {'id': pid, 'type': 'body25'}
        for key in ['keypoints3d', 'handl3d', 'handr3d', 'face3d']:
            if key not in d.keys():continue
            pose3d = np.array(d[key], dtype=np.float32)
            if pose3d.shape[1] == 3:
                pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
            ret[key] = pose3d
        res_.append(ret)
    return res_
def read_keypoints3d_all(path, key='keypoints3d', pids=[]):
    assert os.path.exists(path), '{} not exists!'.format(path)
    results = {}
    filenames = sorted(glob(join(path, '*.json')))
    for filename in filenames:
        nf = int(os.path.basename(filename).replace('.json', ''))
        datas = read_keypoints3d(filename)
        for data in datas:
            pid = data['id']
            if len(pids) > 0 and pid not in pids:
                continue
            # 注意 这里没有考虑从哪开始的
            if pid not in results.keys():
                results[pid] = {key: [], 'frames': []}
            results[pid][key].append(data[key])
            results[pid]['frames'].append(nf)
    if key == 'keypoints3d':
        for pid, result in results.items():
            result[key] = np.stack(result[key])
    return results, filenames
results3d, filenames = read_keypoints3d_all('output-track/keypoints3d/')
