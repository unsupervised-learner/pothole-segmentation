from model import genModel
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#image paths
main_path = 'C:/Users/android18/Pictures/potholes'
image_path = os.path.join(main_path, 'images')
mask_path = os.path.join(main_path, 'binary_masks')
test_im_path = 'C:/Users/android18/Pictures/potholes/test'

#build model
model = genModel()

def train():
	#load images
	train_images = []
	for filename in os.listdir(image_path):
		train_images.append(cv.imread(os.path.join(image_path, filename), 0)/np.max(cv.imread(os.path.join(image_path, filename), 0)))

	#load masks
	train_masks = []
	for filename in os.listdir(mask_path):
		im = cv.imread(os.path.join(mask_path, filename), 0)
		mask1 = im < 215
		mask2 = im >= 215
		im[mask1] = 0
		im[mask2] = 1
		train_masks.append(im)

	print(len(train_images))
	print(len(train_masks))

	#train model
	datagen = ImageDataGenerator()
	it = datagen.flow(np.array(train_images).reshape(-1,128,128,1), np.array(train_masks).reshape(-1,128,128,1))
	#np.array(train_images).reshape(-1,128,128,1), np.array(train_masks).reshape(-1,128,128,1), epochs=1, batch_size=1
	hist = model.fit_generator(it, steps_per_epoch=20)
	#print(hist.history)

	#visualize results
	acc = hist.history['acc']
	loss = hist.history['loss']

	fig = plt.figure(facecolor='black')

	fig.add_subplot(1,2,1)
	plt.plot(acc)

	fig.add_subplot(1,2,2)
	plt.plot(loss)

	plt.show()


def test(testdir):
	
	for filename in os.listdir(testdir):
		test_im = cv.imread(os.path.join(testdir, filename), 0)
		test_im = cv.resize(test_im, (128, 128), cv.INTER_AREA)
		predicted_im = model.predict(test_im.reshape(1,128,128,1))

		plt.imshow(predicted_im.reshape(128,128))
		plt.show()

if __name__ == '__main__':
	#train model
	train()

	#test model
	test(test_im_path)