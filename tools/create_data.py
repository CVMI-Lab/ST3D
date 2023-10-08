# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.data_converter import coda_converter as coda_converter

def coda_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    resolution):
    """Prepare the info file for CODa dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """

    from tools.data_converter import coda_converter as coda

    ### TODO: Set these manually 
    channels=32
    load_dir = osp.join(root_path, 'coda_original_format')
    save_dir = osp.join(out_dir, f'coda{channels}_allclass_full')
    ###
    
    splits = ['training', 'testing', 'validation']
    for split in splits:
        converter = coda.CODa2KITTI(
            load_dir,
            save_dir,
            workers=workers,
            split=split,
            channels=channels
        )
        converter.convert()
        print("length dataset ", len(converter))

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='coda', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data',
    help='specify the root path of dataset. Defaults to ./data')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for coda')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data',
    required=False,
    help='output directory of kitti formatted dataset. Defaults to ./data')
parser.add_argument('--extra-tag', type=str, default='coda')
parser.add_argument('--resolution', type=int, default=32)
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'coda':
        coda_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            resolution=args.resolution
        )
