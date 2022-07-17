from threading import Thread
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

class VideoStream:
  def __init__(self,res,fr):
    self.camera = PiCamera()
    self.camera.resolution = res
    self.camera.framerate = fr
    self.rawCapture = PiRGBArray(self.camera,size = res)
    self.stream = self.camera.capture_continuous(self.rawCapture, format = "bgr", use_video_port = True)
    self.frame = []
    self.stop = False
    
  def start(self):
    Thread(target=self.update, args=()).start()
    return self
  
  def update(self):
    for f in self.stream:
      self.frame = f.array
      self.rawCapture.truncate(0)
      if self.stop:
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()
        
  def read(self):
    return self.frame
  
  def stop(self):
    self.stopped = True
