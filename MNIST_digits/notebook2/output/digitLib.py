import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class DigitModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('saved_model/my_model')
    def predict(self,image):
        return np.argmax(self.model.predict(image))

class DigitCanvas:
    def __init__(self):
        self.image = np.zeros((28,28)) 
        plt.imshow(self.image)
        self.axes = plt.gca()
        self.figure = plt.gcf()
        self.press = None
        self.connect()
	
    def connect(self):
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
#         self.figure.suptitle("New drawing created")
        self.press = 1
        self.image[:,:]=0
        self.axes.imshow(self.image)

    def on_motion(self, event):
        if self.press is None: return
        x=int(event.xdata)
        y=int(event.ydata)
#         self.figure.suptitle(str(x)+" "+str(y))
        self.image[y:y+2,x:x+2]=1
        self.axes.imshow(self.image)

    def on_release(self, event):
        self.press = None
#         self.figure.suptitle("Finished drawing")
    
    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)

    def getDrawing(self):
        return self.image.reshape(1,28,28,1)
