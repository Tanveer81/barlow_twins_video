import time
from glob import glob
import os
import random
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import cv2


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

        number_of_pairs = int(len(img_paths) / frame_dist) * 100
        attempts = 3 * number_of_pairs
        while True:
            start = random.randint(0, len(img_paths) - frame_dist - 1)
            end = random.randint(start + frame_dist, len(img_paths) - 1)
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
    with open(f'{out_path}barlow_twins_pairs.txt', 'w') as fp:
        fp.write('\n'.join('%s %s' % x for x in frame_pairs))


def debug_barlow_twin_annotations(pair_meta_path, img_path):
    start_time = time.time()
    file = open(f'{pair_meta_path}barlow_twins_pairs.txt', 'r')
    pairs = []
    for line in file:
        frames = line.split(' ')
        pairs.append(frames)
    print(f'Total Time for loading Pairs : {(time.time() - start_time)} seconds')
    for i in range(10):
        i = random.randint(0, len(pairs) - 1)
        image = Image.open(os.path.join(img_path, pairs[i][0]))
        image2 = Image.open(os.path.join(img_path, pairs[i][1].rstrip()))
        visualize(image)
        time.sleep(0.5)
        visualize(image2)
        time.sleep(1)


def save_bounding_boxes_for_barlow_twins(path):
    start_time = time.time()
    bboxes_frame = {}
    with open(f'{path}detectron2-annotations-train-balanced.json') as json_file:
        detectron_data = json.load(json_file)
    length = len(detectron_data['annos'])
    for i, frame in enumerate(detectron_data['annos']):
        frame_key = frame['video_id'] + '_' + frame['frame_id']
        bboxes_anno = {}
        for anno in frame['annotations']:
            anno_key = str(anno['category_id']) + '_' + anno['object_id']
            bboxes_anno[anno_key] = anno['bbox']
        bboxes_frame[frame_key] = bboxes_anno
        print(f'{i + 1}/{length} : {(time.time() - start_time)} seconds')

    print(f'Total Time : {(time.time() - start_time)} seconds')
    print(f'Saving bboxes as {path}/barlow_twins_bboxes.json')
    with open(f'{path}/barlow_twins_bboxes.json', 'w') as outfile:
        json.dump(bboxes_frame, outfile)


def is_empty(bboxes_data, pair):
    # Returns true if a pir of frame does not have  bounding box for common objects
    pair = pair.split(' ')
    box1 = bboxes_data[pair[0].replace('/', '_').split('.')[0]]
    box2 = bboxes_data[pair[1].rstrip().replace('/', '_').split('.')[0]]
    common_bbox_list = list(set(list(box1.keys())) & set(list(box2.keys())))
    return not common_bbox_list


def refine_barlow_pairs_boxes(path):
    # remove frame pair with empty bboxes
    start_time = time.time()
    with open(f'{path}barlow_twins_bboxes.json') as json_file:
        bboxes_data = json.load(json_file)
    file = open(f'{path}barlow_twins_pairs.txt', 'r')
    pairs = [line for line in file]
    refined_pairs = [pair for pair in pairs if not is_empty(bboxes_data, pair)]
    print(f'Total Time : {(time.time() - start_time)} seconds')
    print(f'Saving pairs as {path}barlow_twins_pairs_refined.txt')
    with open(f'{path}barlow_twins_pairs_refined.txt', 'w') as fp:
        for row in refined_pairs:
            fp.write(str(row))


def debug_refined_barlow_pairs_boxes(path):
    # remove frame pair with empty bboxes
    start_time = time.time()
    with open(f'{path}barlow_twins_bboxes.json') as json_file:
        bboxes_data = json.load(json_file)
    file = open(f'{path}barlow_twins_pairs_refined.txt', 'r')
    pairs = [line for line in file]
    refined_pairs = [pair for pair in pairs if not is_empty(bboxes_data, pair)]
    print(f'Total Time : {(time.time() - start_time)} seconds')
    # print(f'Saving pairs as {path}barlow_twins_pairs_refined.txt')
    # with open(f'{path}barlow_twins_pairs_refined.txt', 'w') as fp:
    #     for row in refined_pairs:
    #         fp.write(str(row))


def visualize(img, cmap='binary'):
    plt.imshow(img, cmap=cmap)
    plt.show(block=True)


def visualize_bbox(image, bbox):
    # image = copy.deepcopy(image)
    image = np.ascontiguousarray(image)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    visualize(image)


class YvosDateset(Dataset):
    def __init__(self, meta_path, img_path, transform, crop=False, increase_small_area=False):
        super().__init__()
        self.increase_small_area = increase_small_area
        self.crop = crop
        if crop:
            with open(f'{meta_path}barlow_twins_bboxes.json') as json_file:
                self.bboxes_data = json.load(json_file)
        file = open(f'{meta_path}barlow_twins_pairs_refined.txt', 'r')
        self.transform = transform
        self.img_path = img_path
        start_time = time.time()
        self.pairs = []
        for line in file:
            frames = line.split(' ')
            self.pairs.append(frames)
        print(f'Total Time for loading Pairs : {(time.time() - start_time)} seconds')

    def is_small_object(self, box):
        return ((box[2] - box[0]) * (box[3] - box[1])) < 1024

    def add_gaussian_noise(self, size):
        col, row = size
        # Gaussian distribution parameters
        mean = 0
        var = 1
        sigma = var ** 0.5
        gaussImage = np.random.normal(mean, sigma, (row, col)).astype('float32')[..., None]
        gaussImage = (gaussImage - np.min(gaussImage)) / (np.max(gaussImage) - np.min(gaussImage))
        gaussImage = gaussImage * 255
        gaussImage = gaussImage.astype('uint8')
        gaussImage = np.concatenate((gaussImage, gaussImage, gaussImage), axis=2)
        return gaussImage

    def add_random_noise(self, size):
        col, row = size
        gaussImage = np.random.random((row, col))
        gaussImage = (gaussImage * 255 / np.max(gaussImage)).astype('uint8')[..., None]
        gaussImage = np.concatenate((gaussImage, gaussImage, gaussImage), axis=2)
        return gaussImage

    def increase_area(self, box, img_size, scale=2, all_small=False):
        dist_x = (box[2] - box[0])
        dist_y = (box[3] - box[1])

        if all_small:
            if dist_x < img_size[0] / 3:
                scale_x = (img_size[0] / 3) / dist_x
            if dist_y < img_size[1] / 3:
                scale_y = (img_size[1] / 3) / dist_y
        else:
            scale_x = scale
            scale_y = scale

        if box[0] - (dist_x * scale_x / 2) < 0:
            box[0] = 0
            box[2] = dist_x * scale_x
        elif box[2] + (dist_x * scale_x / 2) > img_size[0]:
            box[2] = img_size[0]
            box[0] = img_size[0] - dist_x * scale_x
        else:
            box[0] = box[0] - dist_x * (scale_x - 1) / 2
            box[2] = box[2] + dist_x * (scale_x - 1) / 2

        if box[1] - (dist_y * scale_y / 2) < 0:
            box[1] = 0
            box[3] = dist_y * scale_y
        elif box[3] + (dist_y * scale_y / 2) > img_size[1]:
            box[3] = img_size[1]
            box[1] = img_size[1] - dist_y * scale_y
        else:
            box[1] = box[1] - dist_y * (scale_y - 1) / 2
            box[3] = box[3] + dist_y * (scale_y - 1) / 2
        return [int(b) for b in box]

    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def union_of_boxes(self, boxes, img_size, all_small=False):
        x_min = min([x[0] for x in boxes])
        x_max = max([x[2] for x in boxes])
        y_min = min([x[1] for x in boxes])
        y_max = max([x[3] for x in boxes])
        box = [x_min, y_min, x_max, y_max]
        image_area = (img_size[0] * img_size[1])
        box_area = self.box_area(box)
        if box_area < 0.33 * image_area:
            scale = image_area / box_area
            scale = np.sqrt(scale / 3)
            box = self.increase_area(box, img_size, scale, all_small)
        return box

    def center_bbox(self, box, img_size):
        img_center = (img_size[0] / 2, img_size[1] / 2)
        box_center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)
        x_dist = img_center[0] - box_center[0]
        y_dist = img_center[1] - box_center[1]
        box[0] += x_dist
        box[2] += x_dist
        box[1] += y_dist
        box[3] += y_dist
        return [int(b) for b in box]

    def cut_image(self, image, boxes):
        image_size = image.size
        boxes_temp = [box for box in boxes if not self.is_small_object(box)]
        all_small = len(boxes_temp) == 0
        if all_small:
            gaussImage = image.filter(ImageFilter.GaussianBlur(np.random.randint(20, 50)))
            union_box = self.union_of_boxes(boxes, image_size, all_small)
            image = image.crop(tuple(union_box))
            image = np.array(image)
            gaussImage = np.array(gaussImage)
            gaussImage[union_box[1]:union_box[3], union_box[0]:union_box[2]] = image
            gaussImage = Image.fromarray(gaussImage)
            # visualize(gaussImage)
            return gaussImage
        else:
            boxes = boxes_temp
            union_box = self.union_of_boxes(boxes, image_size)
            rand = random.random()
            if self.box_area(union_box) < 0.5 * (image_size[0] * image_size[1]) and rand > 0.66:
                image = image.crop(tuple(union_box))
                return image
            else:
                if rand <= 0.33:
                    gaussImage = image.filter(ImageFilter.GaussianBlur(np.random.randint(20, 50)))
                    gaussImage = np.array(gaussImage)
                elif 0.33 < rand <= 0.495:
                    gaussImage = self.add_gaussian_noise(image.size)
                else:
                    gaussImage = self.add_random_noise(image.size)

                image = np.array(image)
                # box = self.center_bbox(box, image_size)
                for box in boxes:
                    gaussImage[box[1]:box[3], box[0]:box[2]] = image[box[1]:box[3], box[0]:box[2]]
                gaussImage = Image.fromarray(gaussImage)
            return gaussImage

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image1 = Image.open(os.path.join(self.img_path, pair[0]))
        image2 = Image.open(os.path.join(self.img_path, pair[1].rstrip()))
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        # visualize(image1)
        # visualize(image2)
        box2 = self.bboxes_data[pair[1].rstrip().replace('/', '_').split('.')[0]]
        image2 = self.cut_image(image2, box2.values())
        # visualize(image1)
        # visualize(image2)
        image1, image2 = self.transform(image1, image2)
        # visualize(image1.permute(1, 2, 0))
        # visualize(image2.permute(1, 2, 0))
        return image1, image2

    def __len__(self) -> int:
        return len(self.pairs)


def main():
    root = '/nfs/data3/koner/data'
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    pair_meta_path = f'{root}/youtubeVOS/train/'
    meta_path = f'{root}/youtubeVOS/train/train-train-meta-balanced.json'
    frame_dist = 5
    # generate_barlow_twin_annotations(img_path, meta_path, pair_meta_path, frame_dist)
    # debug_barlow_twin_annotations(pair_meta_path, img_path)
    # save_bounding_boxes_for_barlow_twins(pair_meta_path)
    # refine_barlow_pairs_boxes(pair_meta_path)
    debug_refined_barlow_pairs_boxes


if __name__ == '__main__':
    main()
