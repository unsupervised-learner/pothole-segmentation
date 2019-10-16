#testing streamlit
import pandas as pd
import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time


im_path = 'C:/Users/android18/Pictures/potholes/images'
mask_path = 'C:/Users/android18/Pictures/potholes/extracted_masks'


def load_training_data():
	#load images
	train_images = []
	for imagefile in os.listdir(im_path):
		im = cv.imread(os.path.join(im_path, imagefile), 0)
		train_images.append(im)

def main():
	st.title('Pothole segmentation dashboard')
	st.subheader('Visualize training data')
	im_selected = st.selectbox('select an image', os.listdir(im_path))
	image = cv.imread(os.path.join(im_path, im_selected))
	mask = cv.imread(os.path.join(mask_path, im_selected))

	fig = plt.figure()
	fig.add_subplot(1,2,1)
	plt.imshow(image)
	plt.axis('off')
	plt.title('image', fontsize=15)
	fig.add_subplot(1,2,2)
	plt.imshow(mask)
	plt.title('mask', fontsize=15)
	plt.axis('off')
	st.pyplot()

	st.subheader('Build model')
	st.text('Model information')
	st.write('	> input size: 128 x 128 x 1')
	st.text('Modify training')
	fitness_function = st.selectbox('select fitness function', ('IoU', 'MSE', 'binary_crossentropy', 'categorical_crossentropy'))
	num_epochs = st.selectbox('number of epochs', (1, 3, 10, 100))
	batch_size = st.selectbox('batch size', (1, 100))

	#build model
	if st.button('Retrain model'):
		with st.spinner('loading images'):
			time.sleep(5)
		st.success('images loaded')

		with st.spinner('loading masks'):
			time.sleep(5)
		st.success('mask loaded')
		
	
	
		


if __name__ == '__main__':
	try:
		main()
	except RuntimeError:
		main()