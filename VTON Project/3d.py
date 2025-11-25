import cv2
import numpy as np

img = cv2.imread("tshirt.jpg")
if img is None:
    raise FileNotFoundError("image.jpg not found.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Depth using lightweight stereo trick
shift_pixels = 8
shifted = np.roll(gray, shift_pixels, axis=1)

stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
depth = stereo.compute(gray, shifted).astype(np.float32)

depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)

h, w = gray.shape
anaglyph = np.zeros_like(img)

# BOOST DEPTH × 5
depth_norm = (depth_norm.astype(np.float32) * 5).clip(0, 255).astype(np.uint8)

for y in range(h):
    for x in range(w):
        shift_amt = int((depth_norm[y, x] / 255.0) * 25)  # Stronger shift (25 px)
        new_x = max(0, x - shift_amt)

        anaglyph[y, x, 2] = img[y, new_x, 2]  # Red (shifted)
        anaglyph[y, x, 1] = img[y, x, 1]      # Green
        anaglyph[y, x, 0] = img[y, x, 0]      # Blue

cv2.imwrite("3d_image_strong.png", anaglyph)
print("Saved: 3d_image_strong.png")
