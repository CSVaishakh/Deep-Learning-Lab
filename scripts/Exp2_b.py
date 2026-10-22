import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(gray_img,cv2.MORPH_OPEN,kernel)
closing = cv2.morphologyEx(gray_img,cv2.MORPH_CLOSE,kernel)


plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Color Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray_img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(opening, cmap="gray")
plt.title("Morphological Opening")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(closing, cmap="gray")
plt.title("Morphological Closing")
plt.axis("off")

plt.tight_layout()
plt.show()
