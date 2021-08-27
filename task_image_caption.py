# -*- coding:utf-8 -*-
# ! -*- coding: utf-8 -*-
# bert做image caption任务，coco数据集
# 通过Conditional Layer Normalization融入条件信息
# 请参考：https://kexue.fm/archives/7124

from __future__ import print_function
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, is_string
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from data_preprocess import read_caption_flickr, read_caption_cn, read_image
import warnings

warnings.filterwarnings('ignore')
# 模型配置
maxlen = 64
batch_size = 4
steps_per_epoch = 1000
epochs = 20

# ============================英文=======================================
# # 英文bert配置
# config_path = 'bert/uncased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'bert/uncased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'bert/uncased_L-12_H-768_A-12/vocab.txt'
# # 数据路径
# caption_path = 'data/flickr/flickr30k-caption/results_20130124.token'
# image_path = 'data/flickr/flickr30k-images/'
# # 加载数据
# data = read_caption_flickr(caption_path)
# train_data = data[0:31000]
# valid_data = data[31000:]
# ============================中文=======================================
# 中文bert配置
config_path = 'bert/roberta/bert_config.json'
checkpoint_path = 'bert/roberta/bert_model.ckpt'
dict_path = 'bert/roberta/vocab.txt'
# 数据路径
caption_path = r'D:\Multi-Model Dataset\cn\ai_challenger_caption_train_20170902\caption_train_annotations_20170902.json'
image_path = 'D:/Multi-Model Dataset/cn/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
# 加载数据
data = read_caption_cn(caption_path)
train_data = data[0:200000]
valid_data = data[200000:]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_images, batch_token_ids, batch_segment_ids = [], [], []
        for is_end, D in self.sample(random):
            img = image_path + D['image_id']
            caption = np.random.choice(D['caption'])
            token_ids, segment_ids = tokenizer.encode(caption, maxlen=maxlen)
            batch_images.append(read_image(img))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_images = np.array(batch_images)
                batch_images = preprocess_input(batch_images)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids, batch_images], None
                batch_images, batch_token_ids, batch_segment_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 图像模型
MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
image_model = MobileNetV2(include_top=False, pooling='avg')

# Bert模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    layer_norm_cond=image_model.output,
    layer_norm_cond_hidden_size=128,
    layer_norm_cond_hidden_act='swish',
    additional_input_layers=image_model.input,
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoCaption(AutoRegressiveDecoder):
    """img2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        image = inputs[0]
        token_ids = output_ids
        segment_ids = np.zeros_like(token_ids)
        return self.last_token(model).predict([token_ids, segment_ids, image])

    def generate(self, image, topk=2):
        if is_string(image):
            image = read_image(image)
        image = preprocess_input(image)
        output_ids = self.beam_search([image], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autocaption = AutoCaption(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


def just_show():
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2)]  # 从valid_data中随机取两个
    for D in samples:
        img = image_path + D['image_id']
        print(u'image_id:', D['image_id'])
        print(u'url:', D['url'])
        print(u'predict:', autocaption.generate(img))
        print(u'references:', D['caption'])
        print()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save('model/best_model.model')
        # 演示效果
        just_show()


if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

# else:
#
#     model.load_weights('model/best_model.weights')
