# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:36:14 2020

@author: ramesh
"""
import os
from flask import Flask, render_template, request, session, make_response
import cv2
import numpy as np
import base64
from skimage.measure import compare_ssim


app = Flask(__name__)
app.secret_key = 'abcde'

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

#@app.route('/', methods=['POST'])
#def upload_file():
#    file1 = request.files['image1']
#    npimg = np.fromfile(file1, np.uint8)
#    before = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#    #before = cv2.imread(file1)
#    session['before'] = before
#    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Read image
    file1 = request.files['image1']
    npimg = np.fromfile(file1, np.uint8)
    before = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    before1 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    file2 = request.files['image2']
    npimg = np.fromfile(file2, np.uint8)
    after = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Grey color
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)
    print("Image similarity", score)
    
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
    
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff',diff)
    #cv2.imshow('mask',mask)
    #cv2.imshow('filled after',filled_after)
    cv2.waitKey(0)
    #faces = detect_faces(image)

    
        # Save
        #cv2.imwrite(filename, image)
        
    # In memory
    image_content1 = cv2.imencode('.jpg', before1)[1].tostring()
    encoded_image1 = base64.encodestring(image_content1)
    to_send1 = 'data:before1/jpg;base64, ' + str(encoded_image1, 'utf-8')
    image_content = cv2.imencode('.jpg', after)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:after/jpg;base64, ' + str(encoded_image, 'utf-8')
    

    return render_template('index.html', image_to_show1=to_send1, image_to_show=to_send, init=True)



if __name__ == "__main__":
    # Only for debugging while developing
    app.run(port=5000)
    


