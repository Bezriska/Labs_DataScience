import matplotlib.pyplot as plt
import cv2
import os

frame = cv2.imread("DATA/frames/frame_00059.jpg")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
ax.imshow(rgb)

# При наведении мыши показывает координаты в строке статуса
plt.show()