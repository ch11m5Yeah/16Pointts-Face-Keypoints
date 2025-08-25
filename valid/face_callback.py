import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# ========== 解码预测框 ==========
def integral(distribution):
    """积分模块：将分布回归转为期望值"""
    reg_max = distribution.shape[-1] - 1
    positions = tf.range(reg_max + 1, dtype=tf.float32)
    return tf.reduce_sum(distribution * positions, axis=-1)
def soft_argmax_2d(heatmap, beta=10.0):
    """
    heatmap: [B, H, W, C] 预测的热力图，值越大表示关键点可能性越大
    beta: softmax温度系数，越大越尖锐，越小越平滑

    返回:
        coords: [B, C, 2] 连续坐标，(x,y)格式，坐标范围是 [0, W-1], [0, H-1]
    """
    B = tf.shape(heatmap)[0]
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]
    C = tf.shape(heatmap)[3]

    # reshape为[B, H*W, C]，方便softmax
    flat = tf.reshape(heatmap, [B, H * W, C])
    prob = tf.nn.softmax(flat * beta, axis=1)  # [B, H*W, C]

    # 生成网格坐标
    xs = tf.linspace(0.0, tf.cast(W - 1, tf.float32), W)  # [W]
    ys = tf.linspace(0.0, tf.cast(H - 1, tf.float32), H)  # [H]
    xs, ys = tf.meshgrid(xs, ys)  # [H, W]
    xs = tf.reshape(xs, [-1])     # [H*W]
    ys = tf.reshape(ys, [-1])     # [H*W]

    # 计算期望坐标
    exp_x = tf.reduce_sum(prob * xs[None, :, None], axis=1)  # [B, C]
    exp_y = tf.reduce_sum(prob * ys[None, :, None], axis=1)  # [B, C]

    coords = tf.stack([exp_x, exp_y], axis=-1)  # [B, C, 2]

    return coords


def decode_keypoints(raw_out, conf_threshold=0.5):
    """
    关键点检测解码，将模型输出 [B,12,12,48] 解码成 [B,16,3]
    输出是 (conf, x_norm, y_norm)，坐标归一化0~1。

    Args:
        raw_out: [B,12,12,48]，模型输出
        conf_threshold: 置信度阈值，低于此值的关键点conf置0

    Returns:
        keypoints: [B,16,3]，每个点(conf, x, y)
    """
    B = tf.shape(raw_out)[0]
    H = tf.shape(raw_out)[1]
    W = tf.shape(raw_out)[2]
    num_points = 16

    # reshape成[B,H,W,num_points,3]
    raw_out = tf.reshape(raw_out, [B, H, W, num_points, 3])

    conf = tf.sigmoid(raw_out[..., 0])  # [B,H,W,N]
    dx = tf.tanh(raw_out[..., 1]) * 0.5  # [-0.5,0.5]
    dy = tf.tanh(raw_out[..., 2]) * 0.5

    # 构造网格坐标
    grid_x = tf.range(W, dtype=tf.float32)
    grid_y = tf.range(H, dtype=tf.float32)
    gx, gy = tf.meshgrid(grid_x, grid_y)  # [H,W]
    gx = tf.reshape(gx, [1, H, W, 1])  # [1,H,W,1]
    gy = tf.reshape(gy, [1, H, W, 1])  # [1,H,W,1]

    # 计算每个格子对应的归一化坐标
    x_abs = (gx + dx + 0.5) / tf.cast(W, tf.float32)  # [B,H,W,N]
    y_abs = (gy + dy + 0.5) / tf.cast(H, tf.float32)

    # 对每个关键点维度，在 H*W 网格中找 conf 最大的格子
    conf_reshape = tf.reshape(conf, [B, H * W, num_points])  # [B,H*W,N]
    x_abs_reshape = tf.reshape(x_abs, [B, H * W, num_points])
    y_abs_reshape = tf.reshape(y_abs, [B, H * W, num_points])

    # 找最大置信度对应的位置索引
    max_idx = tf.argmax(conf_reshape, axis=1, output_type=tf.int32)  # [B,N]

    batch_idx = tf.range(B)[:, None]  # [B,1]
    point_idx = tf.range(num_points)[None, :]  # [1,N]
    gather_idx = tf.stack([batch_idx * tf.ones_like(max_idx), max_idx, point_idx * tf.ones_like(max_idx)],
                          axis=-1)  # [B,N,3]

    # 使用tf.gather_nd获取最大置信度的conf,x,y
    # 先把conf_reshape等转成shape [B,H*W,N]方便索引
    batch_indices = tf.reshape(tf.range(B), [B, 1])
    point_indices = tf.reshape(tf.range(num_points), [1, num_points])

    # 用tf.gather_nd选出对应点
    # 需要先把三个张量reshape为[B,H*W,N], 然后根据max_idx取值：
    def gather_along_axis(params, indices):
        # params: [B,H*W,N], indices: [B,N]
        # 对每个batch和关键点，在H*W里根据indices选值
        B_, HW, N_ = tf.shape(params)[0], tf.shape(params)[1], tf.shape(params)[2]
        batch_idx_ = tf.repeat(tf.range(B_), repeats=N_)
        point_idx_ = tf.tile(tf.range(N_), multiples=[B_])
        hw_idx_ = tf.reshape(indices, [-1])
        gather_indices = tf.stack([batch_idx_, hw_idx_, point_idx_], axis=1)
        gathered = tf.gather_nd(params, gather_indices)
        return tf.reshape(gathered, [B_, N_])

    conf_max = gather_along_axis(conf_reshape, max_idx)  # [B,N]
    x_max = gather_along_axis(x_abs_reshape, max_idx)
    y_max = gather_along_axis(y_abs_reshape, max_idx)

    # 置信度阈值过滤
    conf_final = tf.where(conf_max > conf_threshold, conf_max, tf.zeros_like(conf_max))

    keypoints = tf.stack([conf_final, x_max, y_max], axis=-1)  # [B,N,3]
    return keypoints


# ========== IoU 计算 ==========
def bbox_iou(boxes1, boxes2):
    xmin_inter = tf.maximum(boxes1[:, 0], boxes2[:, 0])
    ymin_inter = tf.maximum(boxes1[:, 1], boxes2[:, 1])
    xmax_inter = tf.minimum(boxes1[:, 2], boxes2[:, 2])
    ymax_inter = tf.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = tf.maximum(xmax_inter - xmin_inter, 0.0)
    inter_h = tf.maximum(ymax_inter - ymin_inter, 0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1 + area2 - inter_area
    return inter_area / tf.maximum(union_area, 1e-10)


# ========== 自定义 Callback ==========


class KeypointErrorCallback(keras.callbacks.Callback):
    def __init__(self, val_dataset, log_dir=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir) if log_dir else None

    def on_epoch_end(self, epoch, logs=None):
        total_error = 0.0
        total_visible = 0

        for images, labels in self.val_dataset:
            preds = self.model(images, training=False)   # [B, 1, 1, 48] 或其他形状
            pred_kpts = decode_keypoints(preds)          # [B,16,3]

            # 取置信度阈值过滤，或者用标签mask过滤，这里假设labels含mask在conf通道
            # 只计算可见关键点误差
            gt_conf = labels[..., 0]                      # [B,16]
            gt_xy = labels[..., 1:3]                      # [B,16,2]
            pred_xy = pred_kpts[..., 1:3]                 # [B,16,2]
            tf.print(gt_xy[0][0], pred_xy[0][0])
            # print(pred_xy.shape)
            # print(gt_xy.shape)
            visible_mask = tf.cast(gt_conf > 0.8, tf.float32)  # 视为可见关键点
            error = tf.norm(pred_xy - gt_xy, axis=-1)         # 欧氏距离, [B,16]

            total_error += tf.reduce_sum(error * visible_mask)
            total_visible += tf.reduce_sum(visible_mask)

        mean_error = total_error / (total_visible + 1e-8)
        print(f"\nEpoch {epoch + 1}: Validation Mean Keypoint Error = {mean_error:.4f}")

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("Validation/Mean_Keypoint_Error", mean_error, step=epoch)



def run_tflite_inference(tflite_model_path, test_dataset, num_samples=5):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    input_scale, input_zero_point = input_details[0]['quantization']
    input_dtype = input_details[0]['dtype']

    print(f"[INFO] Input quantization: scale={input_scale}, zero_point={input_zero_point}, dtype={input_dtype}")

    for images, labels in test_dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()

        batch_size = images_np.shape[0]
        num_show = min(num_samples, batch_size)

        for i in range(num_show):
            img = images_np[i]  # shape (H,W,C), float32, range [-1,1]

            # 量化输入
            img_q = img / input_scale + input_zero_point
            img_q = np.round(img_q).astype(np.int32)
            if input_dtype == np.uint8:
                img_q = np.clip(img_q, 0, 255).astype(np.uint8)
            elif input_dtype == np.int8:
                img_q = np.clip(img_q, -128, 127).astype(np.int8)
            else:
                raise ValueError(f"Unsupported input dtype: {input_dtype}")

            input_data = np.expand_dims(img_q, axis=0)

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            raw_out = interpreter.get_tensor(output_index)  # [1, 1, 1, 48] 或类似
            pred_kpts = decode_keypoints(raw_out)[0].numpy()  # [16, 3]

            confs = pred_kpts[:, 0]
            xs = pred_kpts[:, 1]
            ys = pred_kpts[:, 2]

            h, w = img.shape[0], img.shape[1]
            pts_x = (np.round(xs * w)).astype(np.int32)
            pts_y = (np.round(ys * w)).astype(np.int32)

            # 可视化
            img_show = (img + 1.0) / 2.0  # 映射回0~1
            img_show = np.clip(img_show, 0, 1)

            plt.figure(figsize=(8, 6))
            plt.imshow(img_show)

            for j in range(len(confs)):
                c = confs[j]
                if c < 0.3:
                    continue  # 置信度低的点不画
                color = (1 - c, 0, c)  # 蓝-红渐变表示置信度
                plt.scatter(pts_x[j], pts_y[j], s=50, c=[color], marker='o', edgecolors='w', linewidth=1)

            plt.title(f"Sample {i+1} Keypoints")
            plt.axis('off')
            plt.show()
def run_h5_inference(h5_model_path, test_dataset, reg_max, num_samples=5):
    # 加载 H5 模型
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    print(f"[INFO] Loaded model from {h5_model_path}")

    # 随机取一个 batch
    for images, labels in test_dataset.take(1):
        images_np = images.numpy()  # 假设 [0,1] 或 [-1,1]
        labels_np = labels.numpy()

        batch_size = images_np.shape[0]
        num_show = min(num_samples, batch_size)

        plt.figure(figsize=(15, 3 * num_show))

        for i in range(num_show):
            img = images_np[i]

            # 模型预测
            raw_out = model.predict(np.expand_dims(img, axis=0))[0]  # (H, W, C)

            # 解码预测框
            decoded_pred = decode_face_bbox(tf.expand_dims(raw_out, 0), reg_max).numpy()[0]
            conf, xmin, ymin, xmax, ymax = decoded_pred

            # 真实框
            gt_box = labels_np[i, 1:5]

            # 转像素坐标
            h, w = img.shape[0], img.shape[1]
            pred_px = [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
            gt_px = [int(gt_box[0] * w), int(gt_box[1] * h), int(gt_box[2] * w), int(gt_box[3] * h)]

            # 显示图片
            plt.subplot(num_show, 1, i + 1)
            plt.imshow((img * 255).astype(np.uint8))  # 如果 [0,1] 还原为 0-255
            plt.title(f"Sample {i+1}: Conf {conf:.2f}")
            ax = plt.gca()

            # 预测框（红色）
            rect_pred = plt.Rectangle((pred_px[0], pred_px[1]),
                                      pred_px[2] - pred_px[0],
                                      pred_px[3] - pred_px[1],
                                      fill=False, edgecolor='red', linewidth=2, label="Pred")
            ax.add_patch(rect_pred)

            # 真实框（绿色）
            rect_gt = plt.Rectangle((gt_px[0], gt_px[1]),
                                    gt_px[2] - gt_px[0],
                                    gt_px[3] - gt_px[1],
                                    fill=False, edgecolor='green', linewidth=2, label="GT")
            ax.add_patch(rect_gt)
            ax.legend()

        plt.tight_layout()
        plt.show()