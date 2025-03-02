import cv2
import numpy as np

image_names = ['stitch1.jpeg', 'stitch2.jpeg', 'stitch3.jpeg', 'stitch4.jpeg', 'stitch5.jpeg', 'stitch6.jpeg']
sift_obj = cv2.SIFT_create()

images = []
images_with_keypoints = []
keypoints_list = []
descriptors_list = []

for i, name in enumerate(image_names):
    image = cv2.imread(name)
    if image is None:
        print(f"Error loading image: {image}")
        continue
    keypoints, descriptors = sift_obj.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
    images.append(image)
    images_with_keypoints.append(image_with_keypoints)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)
    image_with_keypoints = cv2.resize(image_with_keypoints, (600, 600))
    cv2.imshow(f'Image {i + 1} with keypoints drawn on it', image_with_keypoints)
    cv2.waitKey(0)
    cv2.imwrite(f"Output_Images_panorama/Image_{i + 1}_with_keypoints_drawn_on_it.jpg", image_with_keypoints)

cv2.destroyAllWindows()
stitcher = cv2.Stitcher_create()
status, stitched_image = stitcher.stitch(images)
stitched_image = cv2.resize(stitched_image, (600, 600))
cv2.imshow('Panorama Output', stitched_image)
cv2.waitKey(0)
cv2.imwrite(f"Output_Images_panorama/Panorama.jpg", stitched_image)