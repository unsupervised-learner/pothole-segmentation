from keras import *
import pandas as pd
import numpy as np


def genModel():
	inlayer = layers.Input(shape=(128,128,1))

	c1 = layers.Conv2D(8, kernel_size=(3,3), padding='same', activation='relu')(inlayer)
	c1 = layers.Conv2D(8, kernel_size=(3,3), padding='same', activation='relu')(c1)
	p1 = layers.MaxPool2D((2,2))(c1)

	c2 = layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(p1)
	c2 = layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(c2)
	p2 = layers.MaxPool2D((2,2))(c2)

	c3 = layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(p2)
	c3 = layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(c3)
	p3 = layers.MaxPool2D((2,2))(c3)

	c4 = layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(p3)
	c4 = layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(c4)
	p4 = layers.MaxPool2D((2,2))(c4)

	c5 = layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(p4)
	c5 = layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(c5)

	c6 = layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same')(c5)
	ct1 = layers.concatenate([c4,c6])
	c7 = layers.Conv2D(64, kernel_size=(3,3), padding='same')(ct1)
	c7 = layers.Conv2D(64, kernel_size=(3,3), padding='same')(c7)


	c8 = layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same')(c7)
	ct2 = layers.concatenate([c3,c8])
	c9 = layers.Conv2D(32, kernel_size=(3,3), padding='same')(ct2)
	c9 = layers.Conv2D(32, kernel_size=(3,3), padding='same')(c9)

	c10 = layers.Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same')(c9)
	ct3 = layers.concatenate([c2,c10])
	c11 = layers.Conv2D(16, kernel_size=(3,3), padding='same')(ct3)
	c11 = layers.Conv2D(16, kernel_size=(3,3), padding='same')(c11)

	c12 = layers.Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same')(c11)
	ct4 = layers.concatenate([c1,c12])
	c13 = layers.Conv2D(8, kernel_size=(3,3), padding='same')(ct4)
	c13 = layers.Conv2D(8, kernel_size=(3,3), padding='same')(c13)

	outlayer = layers.Conv2DTranspose(1, kernel_size=(1,1), activation='sigmoid')(c13)

	model = models.Model(input = inlayer, output = outlayer)
	model.compile( optimizer=optimizers.SGD(0.01), metrics = ['accuracy'], loss='binary_crossentropy')
	model.summary()

	return(model)
