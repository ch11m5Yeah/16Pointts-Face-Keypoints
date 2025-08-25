import tensorflow as tf
# from .focal_loss import *
# from .distribution_focal_loss import *

def focal_loss_binary(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    y_true: [B, H*W]
    y_pred: [B, H*W]
    """
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_weight = alpha_factor * tf.pow(1 - pt, gamma)
    loss = -focal_weight * tf.math.log(pt + epsilon)
    return loss

def wing_loss(y_true, y_pred, mask=None, w=2.0, epsilon=2.0):
    x = tf.abs(y_true - y_pred)
    C = w - w * tf.math.log(1 + w / epsilon + 1e-7)

    loss_small = w * tf.math.log(1 + x / epsilon + 1e-7)
    loss_large = x - C

    loss = tf.where(x < w, loss_small, loss_large)  # [B,...]

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        # 保证mask和loss维度广播兼容
        while len(mask.shape) < len(loss.shape):
            mask = tf.expand_dims(mask, axis=-1)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(loss)

    return loss


def combined_wing_l2_loss(y_true, y_pred, mask=None, w=0.1, epsilon=2.0, alpha=0.8):
    C = w - w * tf.math.log(1 + w / epsilon)
    diff = y_pred - y_true
    abs_diff = tf.abs(diff)

    wing_l = tf.where(
        abs_diff < w,
        w * tf.math.log(1 + abs_diff / epsilon),
        abs_diff - C
    )
    l2_l = tf.square(diff)

    loss = alpha * wing_l + (1 - alpha) * l2_l

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        while len(mask.shape) < len(loss.shape):
            mask = tf.expand_dims(mask, axis=-1)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(loss)

    return loss

def generate_map(y_true, height, width, pts_num):
    """
    将归一化关键点标签转成网格形式标签
    输入：
      y_true: [B, pts_num, 3], (conf, x_norm, y_norm)
      height, width: 网格大小 H,W
      pts_num: 关键点数量 N
    输出：
      true_map: [B, H, W, N, 3], 对应 (conf, dx, dy)
    """
    B = tf.shape(y_true)[0]
    H = tf.cast(height, tf.int32)
    W = tf.cast(width, tf.int32)

    true_map = tf.zeros([B, H, W, pts_num, 3], dtype=tf.float32)

    true_conf = y_true[..., 0]   # [B, N]
    true_x_norm = y_true[..., 1] # [B, N]
    true_y_norm = y_true[..., 2] # [B, N]

    # 转换成网格绝对坐标
    true_x_abs = true_x_norm * tf.cast(W, tf.float32)  # [B, N]
    true_y_abs = true_y_norm * tf.cast(H, tf.float32)  # [B, N]

    # 计算对应网格索引
    true_x_cell = tf.cast(tf.floor(true_x_abs), tf.int32)  # [B, N]
    true_y_cell = tf.cast(tf.floor(true_y_abs), tf.int32)  # [B, N]

    # 防止越界
    true_x_cell = tf.clip_by_value(true_x_cell, 0, W - 1)
    true_y_cell = tf.clip_by_value(true_y_cell, 0, H - 1)

    # 计算偏移 dx, dy ∈ [-0.5, 0.5]
    true_dx = true_x_abs - (tf.cast(true_x_cell, tf.float32) + 0.5)  # [B, N]
    true_dy = true_y_abs - (tf.cast(true_y_cell, tf.float32) + 0.5)  # [B, N]

    # 构造 indices 用于 scatter 更新 true_map
    batch_idx = tf.repeat(tf.range(B), repeats=pts_num)          # [B*N]
    y_idx = tf.reshape(true_y_cell, [-1])                        # [B*N]
    x_idx = tf.reshape(true_x_cell, [-1])                        # [B*N]
    point_idx = tf.tile(tf.range(pts_num), multiples=[B])       # [B*N]

    indices = tf.stack([batch_idx, y_idx, x_idx, point_idx], axis=1)  # [B*N, 4]

    # 构造要写入的更新值
    updates = tf.stack([
        tf.reshape(true_conf, [-1]),
        tf.reshape(true_dx, [-1]),
        tf.reshape(true_dy, [-1])
    ], axis=1)  # [B*N, 3]

    # 使用 scatter_nd 更新 true_map
    true_map = tf.tensor_scatter_nd_update(true_map, indices, updates)

    return true_map  # [B, H, W, N, 3]


def face_detector_loss_single_target(num_points=16, alpha=0.25, gamma=2.0):
    """
    YOLO风格稀疏关键点检测loss（支持标签归一化 + 缺失点跳过）
    模型输出: [B, H, W, num_points*3] -> (conf, dx, dy)
    标签: [B, num_points, 3] -> (conf, x_norm, y_norm)
    """
    def loss_fn(y_true, y_pred):
        B = tf.shape(y_pred)[0]
        H = tf.shape(y_pred)[1]  # 网格高度
        W = tf.shape(y_pred)[2]  # 网格宽度

        # === 解析预测值 ===
        true_map = generate_map(y_true, H, W, num_points)
        pred_map = tf.reshape(y_pred, [B, H, W, num_points, 3])

        true_conf = true_map[..., 0]
        pred_conf = tf.sigmoid(pred_map[..., 0])

        loss_conf = focal_loss_binary(true_conf, pred_conf, alpha, gamma)

        pos_mask = tf.cast(true_conf > 0, tf.float32)

        pred_dx = tf.tanh(pred_map[..., 1]) * 0.5
        pred_dy = tf.tanh(pred_map[..., 2]) * 0.5

        true_dx = true_map[..., 1]
        true_dy = true_map[..., 2]

        loss_dx = combined_wing_l2_loss(true_dx, pred_dx, pos_mask)
        loss_dy = combined_wing_l2_loss(true_dy, pred_dy, pos_mask)

        pos_count = tf.reduce_sum(pos_mask)
        pos_ratio = pos_count / tf.cast(B*W*H*num_points, tf.float32)
        pos_ratio = tf.maximum(pos_ratio, 0.01)
        total_loss = loss_conf / pos_ratio + (loss_dx + loss_dy) * 8
        return total_loss

    return loss_fn

if __name__ == "__main__":
    coords = tf.constant(
        [[[1, 0.3, 0.9],
          [0, 0.3, 0.5]],
         [[0, 0.3, 0.5],
          [1, 0.3, 0.5]]
         ]
    )
    print(coords.shape)
    conf = coords[..., 0]
    coord = coords[..., 1:3] * 11
    print(conf.shape, coord.shape)
    y = generate_gaussian_heatmap_with_conf(coord, conf, 12,12,sigma=1.5)
    print(y)
    y_trans = tf.transpose(y, [0, 3, 1, 2])
    y_reshape = tf.reshape(y_trans, [tf.shape(y_trans)[0], tf.shape(y_trans)[1], -1])
    pos_mask = tf.reduce_max(y_reshape, axis=-1) > 0.7
    pos_mask_f = tf.cast(pos_mask, tf.float32)
    print(pos_mask_f)
    print(pos_mask_f.shape)
    r = soft_argmax_2d(y, 10) / 11
    print(r)