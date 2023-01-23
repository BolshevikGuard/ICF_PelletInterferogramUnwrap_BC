import numpy as np
from scipy import ndimage
import cv2   as cv
import os
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects as rso

from getstartpoints import getstartpoints as gsp
from PhaseReduce import PhaseReduce as pr
os.system('cls')

# 读取灰度图
origin = cv.imread('19_middle.png', 0)
plt.figure(1), plt.subplot(221), plt.imshow(origin, cmap='gray'), plt.title('origin')

# 快速傅里叶变换
C = np.fft.fftshift(np.fft.fft2(origin))
plt.subplot(222), plt.imshow(20*np.log(np.abs(C)), cmap='gray'), plt.title('original spectrum')

# 找出直流分量和一级频峰的位置
# 前者在中心，但也写出
Cabs = np.abs(C)
Cpeak = ndimage.maximum_filter(np.abs(C), size=3) # 极大值滤波器寻找局部极值
Cpeak = Cabs[Cabs-Cpeak == 0]
Cpeak = sorted(Cpeak, reverse=True)
[loc0, loc1] = [np.argwhere(Cabs == Cpeak[0])[0], np.argwhere(Cabs == Cpeak[1])[0]]
plt.plot(loc0[1], loc0[0], 'x')
plt.plot(loc1[1], loc1[0], 'o')

d0 = round(np.sqrt((loc0[1]-loc1[1])**2 + (loc0[0]-loc1[0])**2)) # 直流与一级频峰的距离
windowlen = round(np.sqrt(2) * d0)                               # 用作滤波窗的边长
core = np.hanning(windowlen) * np.hanning(windowlen).reshape(-1, 1) # 制造hann窗

# 将hann窗扩展到等大小，并滚动至中心
whole_window = np.r_[core, np.zeros((C.shape[0]-core.shape[0], core.shape[1]))]
whole_window = np.c_[whole_window, np.zeros((C.shape[0], C.shape[1]-core.shape[1]))]
whole_window = np.roll(whole_window, round(C.shape[0]/2-core.shape[0]/2), axis=0)
whole_window = np.roll(whole_window, round(C.shape[1]/2-core.shape[1]/2), axis=1)
plt.subplot(223), plt.imshow(whole_window, cmap='gray')
plt.title('whole hann window')

# 将一级频峰滚动至中心，与hann窗点乘滤波
C = np.roll(C, loc0[0]-loc1[0], axis=0)
C = np.roll(C, loc0[1]-loc1[1], axis=1)
C = C * whole_window
plt.subplot(224), plt.imshow(20*np.log(np.abs(C)+1), cmap='gray'), plt.title('filted spectrum')
plt.show()

# 逆变换后转为幅度角矩阵，再变为uint8——[0:255]
# 边缘检测用Canny凑合代替LoG
Angle = np.angle(np.fft.ifft2(np.fft.ifftshift(C)))
Angle = np.uint8((Angle - Angle.min()) / (Angle.max() - Angle.min()) * 255)
Angle1 = cv.medianBlur(Angle, 3)
plt.figure(2), plt.subplot(121), plt.imshow(Angle, cmap='gray'), plt.title('Angle')
plt.subplot(122), plt.imshow(cv.Canny(Angle, 50, 200), cmap='gray'), plt.title('Angle edge')

# 获取种子点序列
firstpoint = plt.ginput(1)
[startCols, startRows] = gsp(firstpoint[0][1], firstpoint[0][0], \
    rso(cv.Canny(Angle, 250, 300), 300, 2))
plt.plot(startRows, startCols, 'x') # 起始点在轮廓图上叠加显示
plt.show()

EDGE = np.array(cv.Canny(Angle, 50, 200)) != 0 # 配合rso转为bool型
EDGE1 = np.uint8(rso(EDGE, 800, 2))            # 配合cv.dilate转为uint8
plt.figure(3), plt.subplot(121), plt.imshow(EDGE1, cmap='gray'), plt.title('EDGE cleared')

se1 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
se2 = cv.getStructuringElement(cv.MORPH_CROSS,(3, 3))
se3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
EDGE2 = cv.erode(cv.erode(cv.dilate(EDGE1, se1), se3), se2) # 将边缘缺口弥合
plt.subplot(122), plt.imshow(EDGE2, cmap='gray'), plt.title('EDGE strengthened')
plt.show()

unwrapped = pr(Angle, startCols[:8], startRows[:8], EDGE2) # 相位解包裹
plt.figure(4), plt.imshow(unwrapped, cmap='gray'), plt.title('Unwrapped 2D')
plt.show()
