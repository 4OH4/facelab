# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:35:06 2018

@author: Adrian Rosebrock

from: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

Class to frame grab in a separate thread
"""


from threading import Thread
import cv2

class WebcamVideoStream:
    
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
    
		if self.frame is not None:
				# get frame size
				self.h, self.w, self.channels = self.frame.shape
		else:
				raise Exception('Unable to open camera connection')
                
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True