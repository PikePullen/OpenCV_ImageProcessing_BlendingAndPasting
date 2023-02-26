import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# plt.imshow(img1)
# plt.show()

# plt.imshow(img2)
# plt.show()

"""
These images are of different dimensions
Normally we would handle this with masking, 
for now though, will just be distorted
"""

# Convert images to same size
# img1 = cv2.resize(img1, (1200,1200))
# plt.imshow(img1)
# plt.show()

# img2 = cv2.resize(img2, (1200,1200))
# plt.imshow(img2)
# plt.show()

# addWeighted only works on images of the same size
# blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)
# blended = cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.2, gamma=0)
# plt.imshow(blended)
# plt.show()

"""
Overlay a smaller image on top of a larger image
This is essentially a numpy reassignment
"""
# img2 = cv2.resize(img2, (600,600))
# large_img = img1
# small_img = img2
#
# x_offset = 0
# y_offset = 0
#
# x_end = x_offset + small_img.shape[1]
# y_end = x_offset + small_img.shape[0]
#
# # we are basically just swapping pixels from the small image to the large image
# large_img[y_offset:y_end,x_offset:x_end] = small_img
# plt.imshow(large_img)
# plt.show()

"""
Blending images of different sizes
"""
img2 = cv2.resize(img2, (600,600))

# print(img1.shape)
x_offset = 934 - 600
y_offset = 1401 - 600

# print(img2.shape)
rows,cols,channels = img2.shape
# print(rows)
# print(cols)
# print(channels)

"""roi = region of interest"""
roi = img1[y_offset:1401,x_offset:934]

"""prints the cutout of the roi"""
# plt.imshow(roi)
# plt.show()

"""get gray scale"""
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# plt.imshow(img2gray, cmap='gray')
# plt.show()

"""we actually need the inverse of the gray"""
mask_inv = cv2.bitwise_not(img2gray)
# plt.imshow(mask_inv, cmap='gray')
# plt.show()

"""
Unfortunately we dont have color map with this
This basically fills the numpy matrix with 255 values, in the same size as the img2
"""
white_background = np.full(img2.shape, 255, dtype=np.uint8)

"""
Grabs this white_background for all color channels 
then puts the mask on top
now each channel has "do not copy image" in it
"""
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
# plt.imshow(bk)
# plt.show()

"""Transition between three layers"""
# plt.imshow(mask_inv, cmap='gray')
# plt.show()
#
# plt.imshow(img2)
# plt.show()

fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
# plt.imshow(fg)
# plt.show()

"""places the mask and overlay on top of the image slice"""
final_roi = cv2.bitwise_or(roi,fg)
# plt.imshow(final_roi)
# plt.show()

"""
perform actual overlay and masking removing the white background of the image
"""
large_image = img1
small_image = final_roi

large_image[y_offset:y_offset + small_image.shape[0],
            x_offset:x_offset + small_image.shape[1]] = small_image
plt.imshow(large_image)
plt.show()