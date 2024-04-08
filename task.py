import cv2
import numpy as np

def white_patch(image):
    b, g, r = cv2.split(image)

    max_val_b = np.max(b)
    max_val_g = np.max(g)
    max_val_r = np.max(r)

    b_norm = np.divide(b, max_val_b)
    g_norm = np.divide(g, max_val_g)
    r_norm = np.divide(r, max_val_r)

    white_balanced = cv2.merge((b_norm, g_norm, r_norm))

    return white_balanced

def gray_world(image):
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    scale_b = avg_g / avg_b
    scale_r = avg_g / avg_r

    balanced_b = np.multiply(image[:, :, 0], scale_b)
    balanced_r = np.multiply(image[:, :, 2], scale_r)

    if balanced_b.shape == image[:, :, 1].shape == balanced_r.shape:
        gray_balanced = cv2.merge((balanced_b, image[:, :, 1], balanced_r))
    else:
        print("Sizes are not compatible.")

    return gray_balanced

image_path = 'images.jpeg'
image = cv2.imread(image_path)
if image is None:
    print("Failed to load the image. Please check the file path.")
    exit(1)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

white_patch_result = white_patch(image)

gray_world_result = gray_world(image)

cv2.imshow('Original Image', image)
cv2.imshow('White Patch Result', white_patch_result)
cv2.imshow('Gray World Result', gray_world_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
