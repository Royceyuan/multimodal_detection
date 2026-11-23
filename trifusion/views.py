"""Native eight-channel conversion; synchronized raw-pixel crop/flip before normalization."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def one_file(root, stem):
    files=[p for p in Path(root).glob(stem+'.*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')]
    if len(files)!=1:raise ValueError('Missing/ambiguous modality '+stem)
    return files[0]


def load_eight(rgb_path, infrared_root, depth_root, window=None, flip=False):
    rgb_path=Path(rgb_path)
    with Image.open(rgb_path) as im:rgb=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
    with Image.open(one_file(infrared_root,rgb_path.stem)) as im:infrared=np.asarray(im)
    with Image.open(one_file(depth_root,rgb_path.stem)) as im:depth=np.asarray(im)
    if infrared.ndim==3:infrared=cv2.cvtColor(infrared,cv2.COLOR_BGR2GRAY)
    if depth.ndim==3:depth=depth[...,0]
    if rgb.shape[:2]!=infrared.shape[:2] or rgb.shape[:2]!=depth.shape[:2]:raise ValueError('Misaligned image modalities')
    if window is not None:
        x1,y1,x2,y2=window
        if not 0<=x1<x2<=rgb.shape[1] or not 0<=y1<y2<=rgb.shape[0]:raise ValueError('Invalid aligned crop')
        rgb=rgb[y1:y2,x1:x2];infrared=infrared[y1:y2,x1:x2];depth=depth[y1:y2,x1:x2]
    if flip:rgb=rgb[:,::-1].copy();infrared=infrared[:,::-1].copy();depth=depth[:,::-1].copy()
    rgb=rgb.astype(np.float32)/255.0
    infrared=infrared.astype(np.float32)
    if infrared.max(initial=0.0)>1.0:infrared/=255.0 if infrared.max()<=255.0 else max(float(infrared.max()),1.0)
    infrared=infrared[...,None]
    metric=depth.dtype==np.uint16 or depth.max(initial=0)>255
    depth_float=depth.astype(np.float32)
    valid=np.isfinite(depth_float)&(depth_float>=300.0)&(depth_float<=20000.0)
    if not metric:
        valid=np.isfinite(depth_float)&(depth_float>0.0);depth_linear=np.clip(depth_float/255.0,0.0,1.0)
    else:depth_linear=(np.clip(depth_float,300.0,20000.0)-300.0)/19700.0
    depth_linear=np.where(valid,depth_linear,0.0).astype(np.float32)
    inverse_depth=np.where(valid,1.0-depth_linear,0.0).astype(np.float32)
    valid_mask=valid.astype(np.float32);metric_flag=np.full(valid_mask.shape,1.0 if metric else 0.0,dtype=np.float32)
    return np.concatenate([rgb,infrared,depth_linear[...,None],inverse_depth[...,None],valid_mask[...,None],metric_flag[...,None]],axis=2)
