import csv
import cv2

import numpy as np

paths = [
    "data/Road1_fb/driving_log.csv",
    "data/Road1_bf/driving_log.csv",
    "data/Road2_fb/driving_log.csv",
    "data/Road2_bf/driving_log.csv",
]


def get_data_1r():
    x, y = [], []

    for path in paths:
        with open(path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for line in csv_reader:
                img_path = line[0]
                correction = 0

                # We are not using those
                if "left" in img_path:
                    correction = 0.25
                elif "right" in img_path:
                    correction = -0.25

                angle = float(line[3]) + correction
                x.append(cv2.imread(img_path))
                y.append(angle)

    return np.array(x), np.array(y)


def get_data_bins(x, y):
    # Angle is between [-25, 25]
    # Consider:
    # [-25, -5.0) - 0 left
    # [-7.5, 5.0] - 1 center
    # (5, 25] - 2 right

    y *= 25
    new_y = []

    for el in y:
        if el <= -5.0:
            new_y.append(0)
        elif el <= 5:
            new_y.append(1)
        else:
            new_y.append(2)

    # Do nothing with x
    return x, np.array(new_y)
