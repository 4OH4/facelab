# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:35:06 2018

@author: Adrian Rosebrock

from: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

Class to frame grab in a separate thread
"""

from abc import ABC  # , abstractmethod
from threading import Thread
import cv2


class InputStream(ABC):
    def start(self):
        # Actions to perform before reading from the stream
        pass

    def read(self):
        # Get a frame from the stream
        pass

    def stop(self):
        # Stop reading from the stream
        pass

    def release(self):
        # Perform any clean-up
        pass


class WebcamVideoStream(InputStream):
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
            raise Exception("Unable to open camera connection")

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self._update, args=()).start()
        return self

    def _update(self):
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

    def release(self):
        # indicate that the thread should be stopped
        self.stream.release()


class VideoFileStream(InputStream):
    def __init__(self, src: str, downsample=1):
        # initialize the video file and read the first frame
        self.stream = cv2.VideoCapture(src)
        self.downsample = downsample

        if not self.stream.isOpened():
            raise Exception(f"Error opening video file: {src}")
            self.stopped = True
        else:
            self.stopped = False
            # (self.grabbed, self.frame) = self.stream.read()
            self.read()

    def read(self):
        # return the frame most recently read
        if self.stopped or not self.stream.isOpened():
            return None

        # otherwise, read the next frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        if self.frame is not None:
            # get frame size
            h, w, channels = self.frame.shape
            # self.h, self.w, self.channels = self.frame.shape

            if self.downsample != 1:

                h = h // self.downsample
                w = w // self.downsample

                self.frame = cv2.resize(
                    self.frame, (w, h), interpolation=cv2.INTER_AREA
                )

            self.h, self.w, self.channels = h, w, channels

        # else:
        #     raise Exception("Unable to read from video file")

        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        # indicate that the thread should be stopped
        self.stream.release()

    def goToFrame(self, frameNum=0):
        if frameNum != 0:
            raise NotImplementedError

        self.stream.set(2, 0)  # cv2.CV_CAP_PROP_POS_FRAMES