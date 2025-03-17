# implementation of the traditional method of converting image to pixel
# for each frame in the video:
# index colors into a limited palette using K-means clustering
# downsample using an interpolation method like nearest neighbor to get blocky edges
# upscale back to original resolution

import cv2
import numpy as np
import os
import argparse

def image_to_pixel(image_path, output_path, palette_size=8, scale=4):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # crop the image to 256x256
    if h > w:
        image = image[(h - w) // 2:(h + w) // 2, :]
    elif w > h:
        image = image[:, (w - h) // 2:(h + w) // 2]

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    h, w, _ = image.shape

    # k-means clustering
    image_reshaped = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(image_reshaped, palette_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((h, w, 3))

    # downsample
    segmented_image = cv2.resize(segmented_image, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
    segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_NEAREST)

    # upscale
    segmented_image = cv2.resize(segmented_image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # save output
    cv2.imwrite(output_path, segmented_image)

def video_to_pixel(video_frames_folder, result_frames_folder, palette_size=8):
    if not os.path.exists(result_frames_folder):
        os.makedirs(result_frames_folder)
    # process each frame in the video folder, make it end up as 256x256
    for filename in sorted(os.listdir(video_frames_folder)):
        if filename.endswith(".png"):
            input_path = os.path.join(video_frames_folder, filename)
            output_path = os.path.join(result_frames_folder, f"pixel_{filename}")


            image_to_pixel(input_path, output_path, palette_size=palette_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video frames to pixel art")
    parser.add_argument("input", type=str, help="Input video frames folder")
    parser.add_argument("output", type=str, help="Output folder for pixel art frames")
    parser.add_argument("--palette_size", type=int, default=8, help="Number of colors in the palette")
    args = parser.parse_args()

    video_to_pixel(args.input, args.output, palette_size=args.palette_size)