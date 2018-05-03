import os
import sys

import numpy as np

from skimage import io

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python3 reconstruct.py [directory] [foo.png]')
        exit(0)
    else:
        img_path = sys.argv[1]
        image_fp = sys.argv[2]

    images = []

    for f in os.listdir(img_path):
        img = io.imread(os.path.join(img_path, f))

        if img is not None:
            images.append(img.astype(np.float64).flatten())

    img = np.stack(images)
    img_avg = np.mean(img, axis=0)

    ### Calculate eigenvector ###
    img = img.T
    img_avg = img_avg.reshape(-1, 1)
    M = img - img_avg

    print('Doing svd')
    U, s, V = np.linalg.svd(M, full_matrices=False)
    print('SVD Done!')

    ### Reconstruct image ###
    image = io.imread(os.path.join(img_path, image_fp))

    image = image.reshape(-1, 1).astype(np.float64)
    image -= img_avg
    w = np.dot(image.T, U[:,:4])
    rec_img = img_avg + np.sum(w * U[:,:4], axis=1).reshape(-1, 1)

    rec_img -= np.min(rec_img)
    rec_img /= np.max(rec_img)
    rec_img = (rec_img * 255).astype(np.uint8)

    io.imsave('reconstruction.png', rec_img.reshape(600, 600, 3))
