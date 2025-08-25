import os
import json
import tensorflow as tf
import numpy as np
from labelme.utils import img_b64_to_arr
import random
import cv2
# 类别映射
LABEL_MAP = {"mouth_0": 0, "mouth_1": 1, "mouth_2": 2, "mouth_3": 3,
             "right_0": 4, "right_1": 5, "right_2": 6, "right_3": 7, "right_4": 8, "right_5": 9,
             "left_0":10, "left_1":11, "left_2":12, "left_3":13, "left_4":14, "left_5":15
             }


def parse_labelme_json(json_path, target_size=(256, 256)):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data.get('imageData'):
            raise ValueError("缺少 imageData")
        img = img_b64_to_arr(data['imageData'])  # 原始大小 (H, W, 3)
        if img is None:
            raise ValueError("图像解码失败")

        # resize到固定大小
        img = cv2.resize(img, target_size)

        img_height, img_width = img.shape[:2]

        label = np.zeros((16, 3), dtype=np.float32)

        shapes = data.get('shapes', [])
        if len(shapes) == 0:
            return img, label

        for shape in shapes:
            label_name = shape['label']
            if label_name not in LABEL_MAP:
                print(f"跳过未知标签: {label_name} in {json_path}")
                continue
            idx = LABEL_MAP[label_name]
            points = shape['points']
            if len(points) != 1:
                continue
            p = points[0]

            # 归一化相对于resize后的尺寸
            label[idx][0] = 1.0
            label[idx][1] = p[0] / data['imageWidth']
            label[idx][2] = p[1] / data['imageHeight']

        return img, label
    except Exception as e:
        print(f"解析 {json_path} 失败: {e}")
        return None, None


def load_data_from_json(json_files, target_size=(256,256)):
    imgs, labels = [], []
    for file in json_files:
        img, label = parse_labelme_json(file, target_size=target_size)
        if img is None or label is None:
            continue
        imgs.append(img)
        labels.append(label)
    imgs_np = np.array(imgs, dtype=np.uint8)        # 形状 (N, H, W, 3)
    labels_np = np.array(labels, dtype=np.float32)  # 形状 (N, 16, 3)
    return imgs_np, labels_np


def create_tf_face_dataset(json_files, img_size=(224, 224), batch_size=32, shuffle=True):
    """
    使用 from_tensor_slices 创建数据集。
    """
    imgs, labels = load_data_from_json(json_files)
    print(f"加载完成: {len(imgs)} 张图片")

    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

    # 预处理
    def preprocess(img, target):
        img = tf.image.resize(img, [img_size[0], img_size[1]])  # [H, W]
        img = (tf.cast(img, tf.float32) / 255.0 - 0.5) * 2  # [-1, 1]
        return img, target

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(imgs))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def generate_face_datasets(json_dir, input_shape, batch_size,
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"目录 {json_dir} 中没有 JSON 文件")

    random.shuffle(json_files)
    total = len(json_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_files = json_files[:train_size]
    val_files = json_files[train_size:train_size + val_size]
    test_files = json_files[train_size + val_size:]

    print(f"训练集: {len(train_files)} 文件, 验证集: {len(val_files)} 文件, 测试集: {len(test_files)} 文件")

    train_dataset = create_tf_face_dataset(train_files, img_size=input_shape, batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_face_dataset(val_files, img_size=input_shape, batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_face_dataset(test_files, img_size=input_shape, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    json_file = "D:\\Dataset\\faceV3\\Labelss\\2025_06_28_14_13_45_Image_face_0_b40.json"  # 替换成你自己的路径
    img, label = parse_labelme_json(json_file)

    print("关键点置信度和坐标 (conf, x, y)：")
    for i, (conf, x, y) in enumerate(label):
        print(f"点{i}: conf={conf:.2f}, x={x:.4f}, y={y:.4f}")

    plt.imshow(img)
    for i in range(16):
        conf, x, y = label[i]
        if conf > 0.5:
            px = int(x * img.shape[1])
            py = int(y * img.shape[0])
            plt.plot(px, py, 'ro')
            plt.text(px + 2, py - 2, str(i), color='yellow', fontsize=8)
    plt.title("关键点可视化")
    plt.show()