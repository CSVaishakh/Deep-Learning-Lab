import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

plt.figure(figsize=(12,6))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Colour Image")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")

plt.subplot(2, 3, 3)
plt.imshow(equ, cmap='gray')
plt.title("Equalized Image")

plt.subplot(2, 3, 5)
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Grayscale Image Histogram")

plt.subplot(2, 3, 6)
plt.hist(equ.ravel(), 256, [0, 256])
plt.title("Equalized Image Histogram")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
res = np.hstack((gray, equ))
plt.imshow(res, cmap='gray')
plt.title("Grayscale And Equalised Images")
plt.tight_layout()
plt.show()



