import os
import tarfile
import urllib.request
from pathlib import Path
import scipy.io
import shutil
import argparse

def download_oxford_flowers(data_dir=None):
    if data_dir is None:
        data_dir = '/storage/ice-shared/vip-vvk/data/AOT/shared/datasets/oxford_flowers'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading Oxford Flowers-102 to {data_dir}...")
    
    urls = {
        'images': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
        'labels': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        'splits': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
    }
    
    for name, url in urls.items():
        filename = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, filename)
            print(f"    Downloaded {os.path.basename(filename)}")
    
    print("\nExtracting images...")
    images_tar = os.path.join(data_dir, '102flowers.tgz')
    if os.path.exists(images_tar):
        with tarfile.open(images_tar, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("  Images extracted to jpg/")
    
    print("\nOrganizing into train/val folders...")
    labels = scipy.io.loadmat(os.path.join(data_dir, 'imagelabels.mat'))['labels'][0]
    splits = scipy.io.loadmat(os.path.join(data_dir, 'setid.mat'))
    
    train_ids = splits['trnid'][0]
    val_ids = splits['valid'][0]
    
    for split_name, ids in [('train', train_ids), ('val', val_ids)]:
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for img_id in ids:
            label = labels[img_id - 1]
            class_dir = os.path.join(split_dir, f'class_{label:03d}')
            os.makedirs(class_dir, exist_ok=True)
            
            src = os.path.join(data_dir, 'jpg', f'image_{img_id:05d}.jpg')
            dst = os.path.join(class_dir, f'image_{img_id:05d}.jpg')
            
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
        
        print(f"  {split_name}: {len(ids)} images organized into 102 classes")
    
    print(f"\nDataset ready at {data_dir}")
    print(f"  Train: {data_dir}/train")
    print(f"  Val: {data_dir}/val")
    return data_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Oxford Flowers-102 dataset')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory (default: ~/scratch/datasets/oxford_flowers)')
    args = parser.parse_args()
    download_oxford_flowers(data_dir=args.data_dir)
