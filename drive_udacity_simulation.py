import base64
import numpy as np
import socketio
import cv2
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import tensorflow as tf
import cnn_model
import scipy.misc

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


sio = socketio.Server()
app = Flask(__name__)

controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, 'save/model.ckpt')


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"]
        throttle = data["throttle"]
        speed = data["speed"]
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
        image_array = (scipy.misc.imresize(image_array[25:135], [66, 200]) / 255.0)-0.5

        #Calculate new Steering angle based on image input
        steering_angle = cnn_model.y.eval(feed_dict={cnn_model.x: image_array[None, :, :, :], cnn_model.keep_prob: 1.0})[0][0]

        throttle = controller.update(float(speed))

        print("Predicted Angle : " + str(steering_angle))

        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

