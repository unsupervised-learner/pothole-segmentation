#script to preprocess images and masks before building model

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


#path to images
targ_dir = 'C:/Users/android18/Pictures/potholes/images'

#path to masks drawn using segment.py
mask_dir = 'C:/Users/android18/Pictures/potholes/masks'

#path to binary masks
dest_dir = 'C:/Users/android18/Pictures/potholes/binary_masks'


#method to fix naming of images
def fix_im_naming():
	os.chdir(targ_dir)
	images = os.listdir()
	names = np.arange(len(images))
	for index in range(len(images)):
		print(index)
		try:
			#read image
			im = cv.imread(images[index], 0)
			print(im.shape)

			#save image
			plt.imsave(str(index)+'.png', im)
		except AttributeError:
			continue


#method to extract binary mask of image
def binarize_n_save():
	os.chdir(mask_dir)
	masks = os.listdir()
	for mask in masks:
		#read image
		mask_im = cv.imread(mask, 0)

		#create and apply mask
		apply_mask = mask_im < 215
		apply_mask2 = mask_im >= 215
		mask_im[apply_mask] = 0

		#save image
		plt.imsave(os.path.join(dest_dir, mask), mask_im)
		
		



if __name__ == '__main__':

	#fix naming of images
	#fix_im_naming()

	#apply thresholding
	binarize_n_save()

	#resize images
	#resize_n_save(targ_dir, (128,128))

	#resize binary masks
	#resize_n_save(mask_dir, (128,128))