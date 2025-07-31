import cv2
import numpy as np

img = cv2.imread('../img/gaussian_noise.jpg')
'''
# 5x5 평균 필터 커널 생성
kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],])
'''
# 5x5 평균 필터 커널 생성
kernel = np.ones((5,5))/5**2
# 필터 적용
blured = cv2.filter2D(img, -1, kernel)

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured) 
cv2.waitKey()
cv2.destroyAllWindows()
