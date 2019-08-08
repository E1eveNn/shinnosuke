import numpy as np


def im2col(inputs,output_shape,filter_size,stride):
    filter_h, filter_w = filter_size

    # padded size
    _, _, n_H, n_W = output_shape
    batch_nums,n_C_prev=inputs.shape[:2]
    col = np.zeros((batch_nums, n_C_prev, filter_h, filter_w, n_H, n_W))

    for y in range(filter_h):
        y_max = y + stride * n_H
        for x in range(filter_w):
            x_max = x + stride * n_W
            col[:, :, y, x, :, :] = inputs[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_nums * n_H * n_W, -1)
    # store it for bp

    return col


def col2im(inputs_shape,pad_size,filter_size,stride,dcol):
    batch_nums, n_C_prev, n_H_prev, n_W_prev = inputs_shape  # 填充前的shape
    pad_h, pad_w = pad_size

    filter_h, filter_w = filter_size
    n_H = (n_H_prev + 2 * pad_h - filter_h) // stride + 1
    n_W = (n_W_prev + 2 * pad_w - filter_w) // stride + 1

    dcol = dcol.reshape(batch_nums, n_H, n_W, n_C_prev, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    output = np.zeros(
        (batch_nums, n_C_prev, n_H_prev + 2 * pad_h + stride - 1, n_W_prev + 2 * pad_w + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * n_H
        for x in range(filter_w):
            x_max = x + stride * n_W
            output[:, :, y:y_max:stride, x:x_max:stride] += dcol[:, :, y, x, :, :]

    return output[:, :, pad_h:n_H_prev + pad_h, pad_w:n_W_prev + pad_w]