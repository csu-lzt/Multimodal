# def wrapper(f):
#     def wrapper_function(*args, **kwargs):
#         """这个是修饰函数"""
#         return f(*args, **kwargs)
#
#     return wrapper_function
#
#
# @wrapper
# def wrapped():
#     """这个是被修饰的函数"""
#     print('wrapped')
#
#
# print(wrapped.__doc__)  # 输出`这个是修饰函数`
# print(wrapped.__name__)  # 输出`wrapper_function`
import numpy as np

# np.empty:根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
# empty不像zeros一样，并不会将数组的元素值设定为0，因此运行起来可能快一些。在另一方面，它要求用户人为地给数组中的每一个元素赋值，所以应该谨慎使用。
# a = np.empty((1,0), dtype=int)
# print(a)
import cv2
path = 'D:/Multi-Model Dataset/cn/ai_challenger_caption_train_20170902/caption_train_images_20170902/fc658658fb8159d90588013e26366b915c47f339.jpg'
img = cv2.imread(path)
height, width = img.shape[:2]