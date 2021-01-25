#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class cottondisease:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        # load model
        model = load_model('cott_dis.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'diseased cotton leaf'
            return [{ "image" : prediction}]
        elif result[0][1] == 1:
            prediction = 'diseased cotton plant'
            return [{ "image" : prediction}]
        elif result[0][1] == 1:
            prediction = 'fresh cotton leaf'
            return [{ "image" : prediction}]
        else:
            prediction = 'fresh cotton plant'
            return [{"image": prediction}]




