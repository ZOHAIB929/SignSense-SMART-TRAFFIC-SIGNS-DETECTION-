#!/usr/bin/env python

"""
This module contains common utility functions for various samples.
"""

# Python 2/3 compatibility
from __future__ import print_function
import sys

# Check Python version
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2
import os
import itertools as it
from contextlib import contextmanager

# Image file extensions
image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)


def split_filename(filename):
    path, fname = os.path.split(filename)
    name, ext = os.path.splitext(fname)
    return path, name, ext


def squared_norm(a):
    return (a * a).sum(axis=-1)


def norm(a):
    return np.sqrt(squared_norm(a))


def homography_transform(H, x, y):
    xs = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    ys = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    s = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return xs / s, ys / s


def to_rectangle(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)


def rect_to_rect_mtx(src, dst):
    src, dst = to_rectangle(src), to_rectangle(dst)
    scale_x, scale_y = (dst[1] - dst[0]) / (src[1] - src[0])
    trans_x, trans_y = dst[0] - src[0] * (scale_x, scale_y)
    return np.float64([[scale_x, 0, trans_x],
                       [0, scale_y, trans_y],
                       [0, 0, 1]])


def look_at(eye, target, up=(0, 0, 1)):
    forward = np.asarray(target, np.float64) - eye
    forward /= norm(forward)
    right = np.cross(forward, up)
    right /= norm(right)
    down = np.cross(forward, right)
    rotation_matrix = np.float64([right, down, forward])
    translation_vector = -np.dot(rotation_matrix, eye)
    return rotation_matrix, translation_vector


def rotation_matrix_to_vector(R):
    _, u, vt = cv2.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:, 0] * _[0]  # same as np.dot(R, vt[0])
    cosine = np.dot(vt[0], p)
    sine = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(sine, cosine)


def draw_text(dst, position, text):
    x, y = position
    cv2.putText(dst, text, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class Sketcher:
    def __init__(self, window_name, images, colors_function):
        self.prev_point = None
        self.window_name = window_name
        self.images = images
        self.colors_function = colors_function
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def show(self):
        cv2.imshow(self.window_name, self.images[0])

    def on_mouse(self, event, x, y, flags, param):
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_point = point
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_point = None

        if self.prev_point and flags & cv2.EVENT_FLAG_LBUTTON:
            for img, color in zip(self.images, self.colors_function()):
                cv2.line(img, self.prev_point, point, color, 5)
            self.dirty = True
            self.prev_point = point
            self.show()


# Color map data from matplotlib
_jet_data = {
    'red': [(0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)],
    'green': [(0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)],
    'blue': [(0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)]
}

cmap_data = {'jet': _jet_data}


def create_colormap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x + eps]
            yp += [y1, y2]
        channel = np.interp(xs, xp, yp)
        channels.append(channel)
    return np.uint8(np.array(channels).T * 255)


def do_nothing(*args, **kwargs):
    pass


def get_clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


@contextmanager
def Timer(message):
    print(f"{message} ...")
    start_time = get_clock()
    try:
        yield
    finally:
        print(f"{(get_clock() - start_time) * 1000:.2f} ms")


class StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = self.smooth_coef * self.value + (1.0 - self.smooth_coef) * v


class RectangleSelector:
    def __init__(self, window, callback):
        self.window = window
        self.callback = callback
        self.drag_start = None
        self.drag_rect = None
        cv2.setMouseCallback(window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                x0, y0 = np.minimum(self.drag_start, (x, y))
                x1, y1 = np.maximum(self.drag_start, (x, y))
                self.drag_rect = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)

    def draw(self, image):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def dragging(self):
        return self.drag_rect is not None


def chunked(n, iterable, fillvalue=None):
    """Groups elements of iterable into chunks of size n."""
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args) if PY3 else it.izip_longest(fillvalue=fillvalue, *args)


def create_mosaic(width, images):
    """Creates a mosaic of images."""
    images = iter(images)
    img0 = next(images) if PY3 else images.next()
    pad = np.zeros_like(img0)
    images = it.chain([img0], images)
    rows = chunked(width, images, pad)
    return np.vstack(list(map(np.hstack, rows)))


def get_image_size(image):
    return image.shape[1], image.shape[0]


def matrix_dot(*args):
    return reduce(np.dot, args)


def draw_keypoints(image, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(image, (int(x), int(y)), 2, color)
