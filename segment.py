#script to manually segment images ans save as mask

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import os

main_dir = 'C:/Users/android18/Pictures/potholes'
image_dir = os.path.join(main_dir, 'images')
mask_dir = os.path.join(main_dir, 'masks')

def main():
	file = sys.argv[1]
	print('[INFO] collecting {}'.format(file))
	targ_im = os.path.join(image_dir, file)
	image_dims = cv.imread(targ_im,0).shape
	height = image_dims[1]
	width = image_dims[0]
	image = pygame.image.load(targ_im)
	image_matrix = np.zeros((height, width))
	print(image_matrix.shape)


	#create screen
	win = pygame.display.set_mode((height, width))

	while True:
		events = pygame.event.get()
		for event in events:
			#print(event)
			if event.type == pygame.QUIT:
				sys.exit()
			
			if event.type == pygame.MOUSEBUTTONDOWN:
				loc = pygame.mouse.get_pos()
				pygame.draw.rect(image, (255, 255, 255), [loc[0], loc[1], 30, 30])

			if event.type == pygame.KEYDOWN:
				os.system('cls')
				im = pygame.surfarray.array2d(image).T
				plt.imsave(os.path.join(mask_dir, file), im)
				#mask = im[]
				print(im)
				plt.imshow(im)
				plt.show()

		win.blit(image, (0,0) )
		pygame.display.update()


if __name__ == '__main__':
	main()