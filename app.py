from flask import Flask, render_template, request
from resources.model import ReconstructiveSubNetwork,DiscriminativeSubNetwork
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

checkpoint_path='static/models/'
object_name='fabric'
resize_shape=(256,256)


def transform_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if resize_shape != None:
        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
        
    image = image / 255.0
    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)    
    return image


def predict(image_path):
    model = ReconstructiveSubNetwork((256,256,3))
    model_seg = DiscriminativeSubNetwork((256,256,6))
    model.load_weights(checkpoint_path+f'model_{object_name}_weights_{100}.h5')
    model_seg.load_weights(checkpoint_path+f'model_seg_{object_name}_weights_{100}.h5')
    print('Models Loaded')
    gray_batch = transform_image(image_path)
    gray_batch = np.expand_dims(gray_batch, axis=0)
    gray_rec = model(gray_batch)
    joined_in = tf.concat([gray_rec, gray_batch], axis=3)
    out_mask = model_seg(joined_in)
    out_mask=out_mask.numpy()
    # Obtain the coordinates of the bounding box
    out_mask = out_mask[0]
    mask = out_mask[:,:,0]

    # Set threshold value
    threshold = 0.1

    # Create a binary image where anomalous pixels are 1 and non-anomalous pixels are 0
    binary_image = np.where(mask > threshold, 1, 0)

    # Find connected regions of anomalous pixels
    connected_regions, _ =  cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    image=cv2.imread(image_path)
    # image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    original_width=image.shape[1]
    original_height=image.shape[0]
# Draw bounding boxes around the connected regions
    scale_x = original_width / resize_shape[1]
    scale_y = original_height / resize_shape[0]

    for region in connected_regions:
        x, y, w, h = cv2.boundingRect(region)
        if w*h > min_area:
            # Scale the bounding box coordinates
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            cv2.rectangle(image,(x, y),(x+w,y+h),(255,0,0),2)
    cv2.imwrite(f"static/predictions/output.png", image)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    if file:
        # Make sure the file has a valid extension
        if file.filename.split('.')[1].lower() in ['jpg', 'jpeg', 'png', 'gif']:
            # Save the file to the static/uploads folder
            file.save(os.path.join('static/uploads', file.filename))
            # Render the display_image template, passing the file name as a parameter
            # return render_template('display_image.html', image_name=file.filename)
            predict(f'static/uploads/{file.filename}')
            return render_template('display_image.html', image_name='output.png',input_name=file.filename)
        else:
            return render_template('index.html', message='Invalid file type')

if __name__ == '__main__':
    app.run()


