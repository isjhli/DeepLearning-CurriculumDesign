import cv2
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import torch
import numpy as np
import random
from general import non_max_suppression
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.augmentations import letterbox
from camera import *
import time


def process_img(img):
    im = letterbox(img, stride=1)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # 添加一个维度，添加的这个维度指的是批量数
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im



# 关于文件目录的相关操作，保证能把yolov5s-seg.pt调用进来
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# model设置以及设备设置
device = torch.device('cuda:0')
weights = ROOT / 'yolov5s-seg.pt'  # 加载FP32模型
model = DetectMultiBackend(weights=weights, device=device, dnn=False, fp16=True)


def mean_v(data):
    if len(data) > 0:
        mean_value = np.mean(data)
    else:
        mean_value = 0
    return mean_value
    # 深度估计


def get_mask(im):
    # 将图像加载到模型上
    pred, proto = model(im, augment=False)[:2]
    # 应用非极大值抑制
    global mask_test
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000, nm=32)
    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            mask_test = masks[:, :, :].cpu().numpy()
    return mask_test


def draw_label(frame1, height, im, names=model.names):
    s = '%gx%g ' % im.shape[2:]  # print string  # 打印标签
    # 将图像加载到模型上
    pred, proto = model(im, augment=False)[:2]
    # 应用非极大值抑制
    global mask_test, annotator
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000, nm=32)

    for i, det in enumerate(pred):
        predict = np.ascontiguousarray(frame1)
        annotator = Annotator(predict, line_width=1, example=str(model.names))  # 继承对象 用来绘制
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            # 打印mask的数据类型
            # print('mask_type:', type(masks))
            # 显示masks数据的张量结构：
            # print('mask_shapes', masks.shape)
            # 显示单个目标的mask结果
            mask_test = masks[:, :, :].cpu().numpy()
            # print(mask_test.shape)
            # cv2.imshow('mask',mask_test)
            # white_pixels = np.argwhere(mask_test == [255, 255, 255])
            # print('size:', white_pixels.shape)
            # print('details', white_pixels)

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], predict.shape).round()  # rescale boxes to im0 size
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class 每一个类的检测
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到标签里
            # Mask plotting 绘制掩膜 这一句其实还可以改，后边我再看看
            retina_masks = False
            annotator.masks(masks, colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(predict, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                                0).contiguous() / 55 if retina_masks else im[i])
            # 绘制bounding box
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                # if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f} {height[i]:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                # if save_crop:
                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    return annotator.result()


def process_invalid(depth):
    where_are_inf = np.isinf(depth)
    depth[where_are_inf] = np.nan
    where_are_nan = np.isnan(depth)
    depth[where_are_nan] = 0
    return depth


def cut(mask, disp, sub=20, i=2):
    # 计算三维坐标数据值
    # # threeD返回值中的通道1应该就是高度信息
    threeD = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True) * 16
    threeD = process_invalid(threeD)
    # 返回左上角的坐标点(x,y)以及对应的矩形框宽(w)高(h)，传入的数据应该是整型
    # x, y, w, h = cv2.boundingRect(cv2.convertScaleAbs(mask[i, :, :]) for i in range(mask.shape[0]))
    coordinate = []
    for j in range(mask.shape[0]):
        coordinate.append(np.array(cv2.boundingRect(cv2.convertScaleAbs(mask[j, :, :]))))
    coordinate = np.array(coordinate)  # (n,[x,y,w,h])
    # 对指定剖分区域的像素数据进行处理
    x = coordinate[:, 0]
    y = coordinate[:, 1]
    w = coordinate[:, 2]
    h = coordinate[:, 3]

    sub_h = (h / sub).astype(int)
    # 提取指定小矩阵之中的元素高度信息
    # threeD: [480, 640]
    p = mask * threeD[:, :, 1]  # p: [mask[0],480,640]
    rec1 = []

    for j in range(p.shape[0]):
        a = p[j][y[j] + sub_h[j] * (i - 1):y[j] + sub_h[j] * i, x[j]:x[j] + w[j]]
        b = p[j][y[j] + h[j] - sub_h[j] * i:y[j] + h[j] - sub_h[j] * (i - 1), x[j]:x[j] + w[j]]
        rec1.append((mean_v(b) - mean_v(a)) / 1000)
    rec = np.array(rec1)
    # 返回估计的高度并且进行单位转换
    return rec, p


def pre_depth(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rectified = cv2.remap(img, left_map1, left_map2, cv2.INTER_LINEAR)
    return img_rectified


def depth(frame1, frame2, blockSize=8, img_channels=3):
    frame1_rectified = pre_depth(frame1)
    frame2_rectified = pre_depth(frame2)
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(frame1_rectified, frame2_rectified)
    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    return dis_color, disparity


def main():
    # 读取相机数据
    capture = cv2.VideoCapture(0)
    # 计时
    num_frames = 0
    # 开始计时

    # 设置相机分辨率
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度 1280分辨率
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度 480分辨率

    # 预热，获取一帧
    ret, frame = capture.read()
    # 初始化帧率
    fps = 0.0
    while ret:
        # 开始计时
        t1 = time.time()

        # 继续获取下一帧
        capture.grab()
        ret, frame = capture.retrieve()

        # 分割图像
        frame1 = frame[0:480, 0:640]
        frame2 = frame[0:480, 640:1280]  # print('fram1：', frame1.shape)  # fram1： (480, 640, 3)
        im = process_img(frame1)  # print('im：', im.shape)  # im： (3, 480, 640)

        # 获取到检测对象的掩膜
        mask_test = get_mask(im)
        # print('mask_test.shape[0]', mask_test.shape[0])

        # 进行深度匹配，获取数目图像的深度谱
        dis_color, disparity = depth(frame1, frame2)

        # 用掩膜二值谱分割深度谱，获取目标对象的可用深度信息
        height, p = cut(mask_test, disparity, sub=20, i=2)
        # print('height_size:', height.shape[0])
        # print('height[0]:', height[0])

        # 添加标签
        result = draw_label(frame1, height, im)

        # 添加帧率的具体数据
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(frame1, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow('frame', result)
        cv2.imshow('mask_test', mask_test[0])
        cv2.imshow('left', frame1)  # 左侧图像显示
        num_frames += 1

        # 程序终止
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
