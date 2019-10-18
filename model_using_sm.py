from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from segmentation_models.losses import bce_jaccard_loss
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pickle




im_path = 'C:/Users/android18/Pictures/potholes/images'
mask_path = 'C:/Users/android18/Pictures/potholes/extracted_masks'
test_path = 'C:/Users/android18/Pictures/potholes/test'
save_pred_to = 'C:/Users/android18/Pictures/potholes/predictions'


def train():
	#load images
	images = []
	for image in os.listdir(im_path):
		imi = cv.imread(os.path.join(im_path, image))
		images.append(imi)

	#load masks
	masks = []
	for mask in os.listdir(mask_path):
		mask_in = cv.imread(os.path.join(mask_path, mask), 0)
		ret_val, threshed_mask = cv.threshold(mask_in, 37, 1, cv.THRESH_BINARY)
		masks.append(threshed_mask)





	model = Unet('resnet34', encoder_weights='imagenet', input_shape=(128,128,3))
	model.compile('Adam', loss=bce_jaccard_loss, metrics = [iou_score, 'accuracy'])
	model.summary()
	hist = model.fit(x=np.array(images).reshape(-1,128,128,3), y=np.array(masks).reshape(-1,128,128,1), batch_size=10, epochs=15)

	#save model
	filename = 'trained_model.h5'
	model.save(filename, include_optimizer=False)
	#pickle.dump(model, open(filename, 'wb'))


def test():
	#load model from disk
	model = load_model('trained_model.h5')
	model.summary()
	#load testing images
	test_ims = []
	for test_im  in os.listdir(test_path):
		test_image = cv.imread(os.path.join(test_path, test_im))
		predicted_mask = model.predict(np.array(test_image).reshape(-1,128,128,3))
		plt.imsave(os.path.join(save_pred_to, test_im), predicted_mask.reshape(128,128))
		#convert mask to grayscale
		to_test = cv.cvtColor(cv.imread(os.path.join(save_pred_to, test_im)), cv.COLOR_BGR2GRAY)

		#threshold
		ret, threshed_mask = cv.threshold(to_test, 200, 1, cv.THRESH_BINARY)

		#find contours
		images, contours, hierarchy = cv.findContours(threshed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		#draw bounding box
		for c in contours:
			x, y, w, h = cv.boundingRect(c)
			cv.rectangle(test_image, (x,y), (x+w, y+h), (0,0,255), 1)
		#save results
		plt.imsave(os.path.join(save_pred_to, test_im), test_image)
		
		#display results
		plt.imshow(test_image)
		plt.axis('off')
		plt.show()

def main():
	#train()
	test()




if __name__ == '__main__':
	main()