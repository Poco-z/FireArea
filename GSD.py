import cv2
import numpy as np
import os
import csv
from pathlib import Path

def calculate_gsd_area_matrix(H, W, sensor_height=5.7, sensor_width=7.6,
                         focal_length=6.8, flying_height=130, pitch_angle=-42):
    pixel_size_height = sensor_height / H
    pixel_size_width = sensor_width / W
    GSD_h = (flying_height * pixel_size_height) / focal_length
    GSD_w  = (flying_height * pixel_size_width) / focal_length

    VFOV = 2 * np.arctan(sensor_height / (2 * focal_length))
    HFOV = 2 * np.arctan(sensor_width / (2 * focal_length))

    theta_v = ((np.arange(H) - H / 2) / H) * np.degrees(VFOV) + pitch_angle
    theta_h = ((np.arange(W) - W / 2) / W) * np.degrees(HFOV)

    row_gsd = GSD_h / np.cos(np.radians(theta_v))
    col_gsd = GSD_w  / np.cos(np.radians(theta_h))

    row_gsd[(theta_v > 90) | (theta_v < -90)] = np.nan
    col_gsd[(theta_h > 90) | (theta_h < -90)] = np.nan

    gsd_area_matrix = np.outer(row_gsd, col_gsd).astype(np.float32)
    return gsd_area_matrix