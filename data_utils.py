from skimage import img_as_float32
import paddle
import cv2
import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class Augmentation:
    def __init__(self, height=512, width=512, height_lim=300, width_lim=300):
        self.height = height
        self.width = width
        self.height_lim = height_lim  # 切割时原图像素为单位
        self.width_lim = width_lim

    def upload(self, img, seg):
        '''Input image should be BGR uint8 format data.'''
        if type(img) == str:
            img = cv2.imread(img)
        if type(seg) == str:
            seg = cv2.imread(seg)
        self.img = Image.fromarray(img)
        self.seg = Image.fromarray(seg)

    def rotate(self):
        angle = random.randint(0, 360)
        self.img = self.img.rotate(angle)
        self.seg = self.seg.rotate(angle)

    def crop(self):
        width, height = self.img.size
        w_begin = random.randint(0, width - 1)  # 因为是闭区间，从0起算，所以要减1
        w_end = random.randint(w_begin, width)
        h_begin = random.randint(0, height - 1)
        h_end = random.randint(h_begin, height)
        img = np.asarray(self.img)
        img = img[h_begin:h_end, w_begin:w_end]
        seg = np.asarray(self.seg)
        seg = seg[h_begin:h_end, w_begin:w_end]
        if img.shape[0] < self.height_lim or img.shape[1] < self.width_lim:
            print('裁剪过小，放弃此次裁剪')
            return
        try:
            self.upload(img, seg)
        except:
            print(img.shape, seg.shape)
            plt.figure()
            plt.imshow(img)
            plt.figure()
            plt.imshow(seg * 255)
            plt.show()

    def padding(self):
        width, height = self.img.size
        height_padding_max = int(height / 8)
        width_padding_max = int(width / 8)
        left_padding = random.randint(0, height_padding_max)
        right_padding = random.randint(0, height_padding_max)
        top_padding = random.randint(0, width_padding_max)
        bottom_padding = random.randint(0, width_padding_max)
        w_new = width + left_padding + right_padding
        h_new = height + top_padding + bottom_padding

        img_pad = np.zeros((h_new, w_new, 3), dtype=np.uint8)
        seg_pad = np.zeros((h_new, w_new, 3), dtype=np.uint8)

        h_begin = top_padding
        h_end = top_padding + height
        w_begin = left_padding
        w_end = left_padding + width
        img_pad[h_begin:h_end, w_begin:w_end] = np.asarray(self.img)
        seg_pad[h_begin:h_end, w_begin:w_end] = np.asarray(self.seg)

        self.upload(img_pad, seg_pad)

    def size_uniform(self):
        self.img = self.img.resize((self.width, self.height))
        self.seg = self.seg.resize((self.width, self.height))

    def augment(self, img=None, seg=None):
        if img is not None and seg is not None:
            self.upload(img, seg)
        self.rotate()
        self.crop()
        self.padding()
        self.size_uniform()
        return np.asarray(self.img), np.asarray(self.seg)

def img_save(imgs, root, prefix='', save_num=None):
    if not os.path.exists(root):
        os.makedirs(root)
    if not isinstance(prefix, str):
        prefix = str(prefix)
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    if save_num is not None:
        imgs = imgs[:save_num]
    img_id = 0
    for img in imgs:
        img_name = prefix+'_'+str(img_id)+'.png'
        img_path = os.path.join(root, img_name)
        Image.fromarray((255*img).astype(np.uint8)).save(img_path)
        img_id += 1

def img2square(img, img_size=512, fill_value=1):
    H,W = img.shape[:-1]
    max_size = max([H, W])
    max_side = np.argmax([H, W])
    another_side = 1 - max_side
    max_new = img_size
    another_new = int([H, W][another_side] * img_size / max_size)
    shape_new = [0, 0]
    shape_new[1 - max_side] = max_new
    shape_new[1 - another_side] = another_new
    img = cv2.resize(img, tuple(shape_new))
    img_new = fill_value*np.ones((img_size, img_size, 3))
    img_new[:img.shape[0], :img.shape[1]] = img
    img = img_new
    return img

def calculate_steps(root, batch_size=16):
    imgs_list = os.listdir(root)
    imgs_num = len(imgs_list)
    return imgs_num//batch_size+1

def train_generator(root, batch_size=16, resize_dst=(512, 512)):
    while True:
        imgs_list = os.listdir(root)
        imgs_list = [os.path.join(root, img_name) for img_name in imgs_list]
        img_batch = []
        for img_path in imgs_list:
            img = cv2.imread(img_path)
            img = img_as_float32(img)
            if img.shape[:-1] != resize_dst:
                img = img2square(img, img_size=resize_dst[0])
            img = img.transpose((2, 0, 1))
            img_batch.append(img)
            if len(img_batch) == batch_size or img_path == imgs_list[-1]:
                img_batch = np.array(img_batch)
                img_batch = paddle.to_tensor(img_batch, dtype=np.float32)
                yield img_batch
                img_batch = []
        yield None

def dataset_resize(root,resize_dst=(512, 512)):
    imgs_list = os.listdir(root)
    imgs_list = [os.path.join(root, img_name) for img_name in imgs_list]
    with tqdm(total=len(imgs_list), ncols=100) as bar:
        bar.set_description('Dataset Checking')
        for img_path in imgs_list:
            img = cv2.imread(img_path)[..., :3]
            if img.shape[:-1] != resize_dst:
                img = img2square(img, img_size=resize_dst[0], fill_value=255)
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(img).save(img_path)
                print(img_path, 'is resized.')
            bar.update(1)

def clear_output():
    clear_str = ['rm models/*',
                 'rm imgs_generated/*']
    for cstr in clear_str:
        os.system(cstr)
        print(cstr)