import cv2
import numpy as np


def apply_flip(image, steering_angle):
    return np.fliplr(image), -1 * steering_angle


def apply_brightness(image, steering_angle):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + np.random.uniform(-0.5, 0.4))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), steering_angle


def apply_shear(image, steering_angle, shear_range=20):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2.0) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle
