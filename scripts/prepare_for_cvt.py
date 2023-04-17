import os
import numpy as np
import cv2
import argparse
from rich.progress import track
from random import seed,shuffle
from time import time

def safeimread(src_fn, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(src_fn, dtype=np.uint8), flags=flags)


def path(x):
    if os.path.exists(x):
        return x
    raise FileNotFoundError(x)

def mkshp(x):
    return tuple([int(y.strip()) for y in x.split(',')])

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('img_dir', type=path, help='test config file path')
    parser.add_argument('dst_dir', type=str,help='checkpoint file')
    parser.add_argument('-t', type=str,default='tidlmodel', help='checkpoint file')
    parser.add_argument('-n',type=int, default=100,)
    parser.add_argument(
        '--shape',
        type=mkshp,
        default=[640, 640],
        help='input image shape, height, width')
    parser.add_argument('-f', type=str, default='bgr',help='input image format')

    args = parser.parse_args()
    return args

def mkbgr(srcimg_fn, wh):
    srcimg = safeimread(srcimg_fn)
    dst_img = cv2.resize(srcimg, wh,interpolation=cv2.INTER_LINEAR)
    return dst_img


def make_ds_for_tidl_cvt(args):
    seed(time())
    img_dir = f'{args.t}_img'
    cvtted_img_dir = os.path.join(args.dst_dir,img_dir)
    os.makedirs(os.path.join(args.dst_dir,f'{args.t}_img'), exist_ok=True)
    fds = open(os.path.join(args.dst_dir,'dataset.txt'),'w')
    fn_ls = os.listdir(args.img_dir)
    shuffle(fn_ls)
    fn_ls = fn_ls[:args.n]
    for i,fn in track(enumerate(fn_ls)):
        img_path = f'{i:0>4}.png'
        srcimg_fn = os.path.join(args.img_dir, fn)
        dstimg_fn = os.path.join(cvtted_img_dir, img_path)
        wh = (args.shape[1],args.shape[0])
        dst_img = mkbgr(srcimg_fn, wh)
        cv2.imwrite(dstimg_fn, dst_img)
        fds.write(f'{img_dir}/{img_path}\n')
    fds.close()
    return

if __name__ == '__main__':
    args = parse_args()
    make_ds_for_tidl_cvt(args)