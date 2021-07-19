import time
from glob import glob
import os
import random
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

def generate_barlow_twin_annotations(img_path, meta_path, out_path, frame_dist):
    f = open(meta_path, )
    data = json.load(f)
    f.close()
    video_id_names = list(data.keys())
    video_id_names.sort()
    frame_pairs = []
    start_time = time.time()
    for i, video_id in enumerate(video_id_names[:3]):
        img_paths = np.sort(glob(os.path.join(img_path, video_id, '*.jpg'))).tolist()
        if len(img_paths) <= frame_dist:
            continue
        categories = list(data[video_id]['objects'].values())
        frames = [a.split('/')[-1].split('.')[0] for a in img_paths]
        frame_categories = dict((a, []) for a in frames)

        for j, category in enumerate(categories):
            for frame in category['frames']:
                frame_categories[frame].append(j)

        number_of_pairs = int(len(img_paths)/frame_dist) * 100
        attempts = 3 * number_of_pairs
        while True:
            start = random.randint(0, len(img_paths) - frame_dist-1)
            end = random.randint(start + frame_dist, len(img_paths)-1)
            key = f'{video_id}/{frames[start]}.jpg'
            if key not in frame_pairs and set(frame_categories[frames[start]]) == set(frame_categories[frames[end]]):
                frame_pairs.append((key, f'{video_id}/{frames[end]}.jpg'))
                number_of_pairs -= 1
                if number_of_pairs == 0:
                    break
            attempts -= 1
            if attempts == 0:
                break

        print(f'{i + 1}/{len(video_id_names)}: {video_id} : {(time.time() - start_time)} seconds')

    print(f'Total Time : {(time.time() - start_time)} seconds')
    print(f'Saving pairs as {out_path}barlow_twins_pairs.txt')
    '''Total Time : 49643.86153244972 seconds ~ 13.78h
       Saving pairs as /nfs/data3/koner/data/youtubeVOS/train/barlow_twins_pairs.txt'''
    with open(f'{out_path}barlow_twins_pairs_test.txt', 'w') as fp:
        fp.write('\n'.join('%s %s' % x for x in frame_pairs))

def visualize(img, cmap='binary'):
    plt.imshow(img, cmap=cmap)
    plt.show(block=True)

def debug_barlow_twin_annotations(pair_meta_path, img_path):
    start_time = time.time()
    file = open(f'{pair_meta_path}barlow_twins_pairs.txt', 'r')
    pairs = []
    for line in file:
        frames = line.split(' ')
        pairs.append(frames)
    print(f'Total Time for loading Pairs : {(time.time() - start_time)} seconds')
    for i in range(10):
        i = random.randint(0, len(pairs)-1)
        image = Image.open(os.path.join(img_path, pairs[i][0]))
        image2 = Image.open(os.path.join(img_path, pairs[i][1].rstrip()))
        visualize(image)
        time.sleep(0.5)
        visualize(image2)
        time.sleep(1)


class YvosDateset(Dataset):
    def __init__(self, pair_meta_path, img_path, transform, crop=False, meta_path=None):
        super().__init__()
        self.crop = crop
        if self.crop:
            f = open(meta_path, )
            self.meta = json.load(f)
        self.transform = transform
        self.img_path = img_path
        start_time = time.time()
        file = open(f'{pair_meta_path}barlow_twins_pairs_test.txt', 'r')
        self.pairs = []
        for line in file:
            frames = line.split(' ')
            self.pairs.append(frames)
        print(f'Total Time for loading Pairs : {(time.time() - start_time)} seconds')

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = Image.open(os.path.join(self.img_path, pair[0]))
        image2 = Image.open(os.path.join(self.img_path, pair[1].rstrip()))
        image = image.convert('RGB')
        image2 = image2.convert('RGB')
        # visualize(image)
        # visualize(image2)
        image, image2 = self.transform(image, image2)
        # visualize(image.permute(1, 2, 0))
        # visualize(image2.permute(1, 2, 0))
        return image, image2

    def __len__(self) -> int:
        return len(self.pairs)


def main():
    root = '/nfs/data3/koner/data'
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    pair_meta_path = f'{root}/youtubeVOS/train/'
    meta_path = f'{root}/youtubeVOS/train/train-train-meta-balanced.json'
    frame_dist = 5
    # generate_barlow_twin_annotations(img_path, meta_path, detectron2_annos_path, frame_dist)
    debug_barlow_twin_annotations(pair_meta_path, img_path)


if __name__ == '__main__':
    main()
