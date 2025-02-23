# -*- coding: utf-8 -*-
import cv2
import numpy as np
from enum import Enum

from math import ceil, sqrt
from matplotlib import pyplot as plt


class Method(Enum):
    CAM_IMAGE_JET = 0
    CAM_IMAGE_BONE = 1
    CAM_AS_WEIGHTS = 2
    JUST_CAM_JET = 3
    JUST_CAM_BONE = 4


def convert_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cam_image(cam, image_bgr, color_map=cv2.COLORMAP_JET):
    height, width, _ = image_bgr.shape  # Get target shape

    cam = 255 * cam
    cam = cv2.resize(cam, (width, height))  # Resize cam to image shape

    heatmap = cv2.applyColorMap(np.uint8(cam), color_map)

    heatmap = np.uint8(heatmap * 0.5)  # Scale heatmap to sum with image
    image_bgr = np.uint8(image_bgr * 0.5)  # Scale image to sum with heatmap

    return heatmap + image_bgr


def just_cam(cam, shape=(244, 244), color_map=cv2.COLORMAP_JET):
    cam = 255 * cam
    # Resize to input shape using bi-linear interpolation
    cam = cv2.resize(cam, shape)

    heatmap = cv2.applyColorMap(np.uint8(cam), color_map)

    return heatmap


def cam_as_weights(cam, image_bgr):
    height, width, _ = image_bgr.shape  # Get target shape

    weighted_image = np.zeros(shape=(height, width, 3))

    cam = cv2.resize(cam, (width, height))
    weighted_image[:, :, 0] = image_bgr[:, :, 0] * cam
    weighted_image[:, :, 1] = image_bgr[:, :, 1] * cam
    weighted_image[:, :, 2] = image_bgr[:, :, 2] * cam
    return np.uint8(weighted_image)


def create_cam_image(cam, image_rgb, visualize_mode):
    image = convert_to_bgr(image_rgb)
    height, width, _ = image.shape
    shape = (width, height)

    if visualize_mode == Method.CAM_IMAGE_JET:
        return cam_image(cam, image, cv2.COLORMAP_JET)
    elif visualize_mode == Method.CAM_IMAGE_BONE:
        return cam_image(cam, image, cv2.COLORMAP_BONE)
    elif visualize_mode == Method.JUST_CAM_JET:
        return just_cam(cam, shape, cv2.COLORMAP_JET)
    elif visualize_mode == Method.JUST_CAM_BONE:
        return just_cam(cam, shape, cv2.COLORMAP_BONE)
    elif visualize_mode == Method.CAM_AS_WEIGHTS:
        return cam_as_weights(cam, image)


def create_guided_cam_image(cam, image_rgb, cam_rate=0.5):
    image = convert_to_bgr(image_rgb)
    height, width, _ = image.shape
    cam = cv2.resize(cam, (width, height))
    return np.uint8(cam * cam_rate) + np.uint8(image * (1-cam_rate))


def plot(cams, filename):
    n_items = len(cams)
    n_rows = ceil(sqrt(n_items))
    n_columns = ceil(n_items/n_rows)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(3*n_columns, 3*n_rows))
    fig.suptitle("This is how we processed your image.")
    # + " Visualization method: " + cams[0].method)
    axes_ravel = axes.ravel()

    for i in range(n_rows*n_columns):
        axes_ravel[i].axis('off')  # Remove Axis
        if i < n_items:
            axes_ravel[i].imshow(convert_to_rgb(cams[i].image))
            axes_ravel[i].set_title(cams[i].layer)

    fig.subplots_adjust(hspace=0.3)

    return plt
