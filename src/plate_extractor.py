import cv2
import numpy as np

win_name = "scanning"
img = cv2.imread("../img/20모5468-3.jpg")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)