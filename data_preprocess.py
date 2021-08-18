# -*- coding:utf-8 -*-
'''
数据预处理的工具函数
'''
import pandas as pd
import cv2


def read_caption_flickr(f):
    """读取并整理flickr数据集的Caption数据
    """
    data = pd.read_table(f, sep='\t', header=None, names=['image', 'caption'])
    images = {}
    image_id = list(map(lambda x: x[:-6], data['image'].values.tolist()))
    caption = data['caption'].values.tolist()
    id_caption = list(zip(image_id, caption))
    set_image_id = list(set(image_id))
    for id in set_image_id:
        images[id] = {'image_id': id + '.jpg',
                      'caption': []}
    for id_temp, caption_temp in id_caption:
        images[id_temp]['caption'].append(caption_temp)
    return list(images.values())


def read_image(f, img_size=299):
    """单图读取函数（对非方形的图片进行白色填充，使其变为方形）
    """
    img = cv2.imread(f)
    height, width = img.shape[:2]
    if height > width:
        height, width = img_size, width * img_size // height
        img = cv2.resize(img, (width, height))
        delta = (height - width) // 2
        img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=0,
            left=delta,
            right=height - width - delta,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    else:
        height, width = height * img_size // width, img_size
        img = cv2.resize(img, (width, height))
        delta = (width - height) // 2
        img = cv2.copyMakeBorder(
            img,
            top=delta,
            bottom=width - height - delta,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    img = img.astype('float32')
    return img[..., ::-1]  # cv2的读取模式为BGR，但keras的模型要求为RGB


#
if __name__ == "__main__":
    import numpy as np

    data = read_caption_flickr(r'E:\暑期文件传输\多模态数据集\flickr 30k\flickr30k\results_20130124.token')
    train_data = data[0:31000]
    valid_data = data[31000:]

    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2)]
    print(np.random.choice(len(valid_data), 2))
