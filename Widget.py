# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:57:15 2018

@author: Rupert.Thomas

Widgets for facelab
Each widget provides a video overlay that displays a specific feature, e.g. histogram plot, timer etc

All inherit from the Widget class, and must implement the update() method
Their output is read from Widget.output

"""

from common import clock, draw_str
import cv2 as cv2
from threading import Thread
import time
import numpy as np

class Widget:
    # Super-class for image overlay objects
    position = 'TL'
    x = 0
    y = 0
    w = 10
    h = 10
    update_period = 1
    output = None

    def __init__(self, interface, position='TL', size=None):
        """
        Constructor
        :param interface: the top-level gui object
        :param position: location in the image to put the GUI {'TL', 'TR', 'BL', 'BR'}
        """
        self.interface = interface
        if size is not None:
            self.w, self.h = size
        self.position = position
        # Determine pixel position (top-left corner is 0,0) from quadrant position
        if 'T' in position:
            self.y = 0
        else:
            self.y = self.interface.gui_h - self.h
        if 'L' in position:
            self.x = 0
        else:
            self.x = self.interface.gui_w - self.w
            
        # Create output space
        self.reset_output()
        
        # start the thread to read frames from the video stream
        self.running = True
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True # stop if the program exits
        

    def update(self):
        # All child objects must over-ride this method, and write to self.output
        pass
    
    def run(self):
        # Target for threading
        while self.running:
            self.update()
            time.sleep(self.update_period)
    
    def create_output(self):
        self.output = np.zeros((self.h, self.w, 4)).astype(np.uint8)
    
    def reset_output(self):
        self.output = None
        
    def direct_write(self, img):
        # called for every screen refresh - use sparingly!
        pass


class Timer(Widget):
    # Adds an elapsed time to the image
    w = 200
    h = 100
    update_period = .1
    verbose = True
    time_last_frame = clock()
    text_offset_x = 10
    
    def __init__(self, interface, position='TL', size=None):
        super().__init__(interface, position, size)
        
        if 'L' in position:
            self.text_align = 'L'
        else:
            self.text_align = 'R'
            self.text_offset_x = self.w - self.text_offset_x  # sync text alignment with position

    def update(self):
        # Update elapsed time
        self.create_output()        
        dt = clock() - self.time_last_frame
        output_text = 'time: %.1f s' % dt
        draw_str(self.output, (self.text_offset_x, 20), output_text, self.text_align)


class HistogramPlot(Widget):
    # Adds a histogram plot of pixel intensities to the image
    w = 150
    h = 50
    update_period = 1

    plot_alpha = 0.7
    grid = True
    grid_lineThickness = 1
    numBins = 256
    scaling_buffer = []  # Take the moving median scaling factor for smoothness
    scaling_buffer_max_len = 5

    def __init__(self, interface, position='TL', size=None):
        super().__init__(interface, position, size)
        
        self.hist_pts_x = np.linspace(0,1,self.numBins)
        
        if self.grid:  # create coordinates for grid
            self.vert_x = np.arange(self.w/4, self.w, self.w/4, dtype=np.int32)
            self.horiz_y = np.arange(self.h/4, self.h, self.h/4, dtype=np.int32)

    def update(self):
        #self.reset_output()
        
        if self.interface.last_frame is not None:
            img = self.interface.last_frame.copy()
            output_buffer = np.zeros((self.h, self.w, 4)).astype(np.uint8)
            
            if len(img.shape)>2 and img.shape[-1]==4:
                img = img[...,:3]  # remove alpha
                
            # Create histogram plot image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Merge channels
            hist_pts_y = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1)
                        
            # Scale max range with smoothing
            self.scaling_buffer.append(hist_pts_y.max())
            if (len(self.scaling_buffer)>self.scaling_buffer_max_len): 
                self.scaling_buffer = self.scaling_buffer[-self.scaling_buffer_max_len:]
            scale_factor = np.median(self.scaling_buffer)
            hist_pts_y = hist_pts_y / scale_factor
            
            # Smooth plot for clarity
            hist_pts_y = conv_smooth(hist_pts_y, 7)
            
            # Make histogram points into a closed polygon
            hist_pts = np.stack([self.hist_pts_x, hist_pts_y], axis=1)
            hist_pts = np.concatenate([np.array([[0,0]]), hist_pts, np.array([[1,0]]), np.array([[0,0]])], axis=0)
            
            # Scale to fill region, and flip (so low is at bottom)
            hist_pts = (hist_pts * np.array([[self.w, self.h]])).astype(np.int32)            
            cv2.fillPoly(output_buffer,[hist_pts],(255,255,255,255*self.plot_alpha))  
                        
            if self.grid:  # add gridlines to plot
                for _x in self.vert_x:
                    cv2.line(output_buffer, (_x, 0), (_x, self.h), (0,0,0,0), self.grid_lineThickness)
                for _y in self.horiz_y:
                    cv2.line(output_buffer, (0, _y), (self.w, _y), (0,0,0,0), self.grid_lineThickness)            
                   
            self.output = np.flipud(output_buffer)
            

class RingBuffer:
    # A 1D/2Dish ring buffer using numpy arrays
    # https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
    def __init__(self, length, width):
        self.data = np.zeros((length, width), dtype='f')
        self.index = 0  # new data will be written at the index value and onwards

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.shape[0])) % self.data.shape[0]
        self.data[x_index,:] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.shape[0])) % self.data.shape[0]
        return self.data[idx]
    
    def __call__(self):
        return self.get()
    

class FPS_plot(Widget):
    # Adds plot of fps to the image
    # Stores frame rate information in a ring buffer object
    w = 200
    h = 75  # full height, including text
    text_offset_x = 10
    text_offset_y = cv2.getTextSize('test', cv2.FONT_HERSHEY_PLAIN, 1, 2)[0][1] + 4
    plot_h = h - text_offset_y  # plot only - leave gap for text at top
    update_period = 1  # second
    fps = 0.0
    last_frame_seen = 0
    num_fps_points_in_memory = 20
    time_last_fps_update = clock()
    fps_buffer = RingBuffer(length=num_fps_points_in_memory, width=2)
    plot_buffer = np.zeros((plot_h, w, 4)).astype(np.uint8)
    line_width = 2  # np.max((2, h//100))
    axes_lineThickness = 2
    plot_alpha = 0.7
    fps_scale_minor_unit = 10    
    smoothing_MA = 3 
        
    def __init__(self, interface, position='TL', size=None):
        super().__init__(interface, position, size)
                
        if 'L' in position:
            self.text_align = 'L'
        else:
            self.text_align = 'R'
            self.text_offset_x = self.w - self.text_offset_x  # sync text alignment with position

    def update(self):
        # Update frame rate        
        dt2 = clock() - self.time_last_fps_update
        self.fps = (self.interface.frame_count - self.last_frame_seen) / dt2
        self.last_frame_seen = self.interface.frame_count
        self.time_last_fps_update = clock()
        
        # Store new data to ring buffer
        self.fps_buffer.extend(np.array([[clock(), self.fps]]))
        
        # Render the plot
        self.plot_buffer = self.genPlot(self.fps_buffer.get())
        
        # Assemble the output: top half text, bottom half graph
        output_buffer = np.zeros((self.h, self.w, 4)).astype(np.uint8)
        output_buffer[self.h-self.plot_h:,...] = self.plot_buffer.copy()
        
        # Add text for current FPS
        draw_str(output_buffer, (self.text_offset_x, self.text_offset_y-2), 'fps: %.1f' % self.fps, self.text_align)
                                    
        self.output = output_buffer.copy()                      

    def genPlot(self, fps_history):
        # Create graph image

        output_buffer = np.zeros((self.plot_h, self.w, 4)).astype(np.uint8)
        
        y_values = moving_average(fps_history[:, 1], n=self.smoothing_MA)
        x_values = fps_history[-y_values.size:, 0]
        
        # Scale timestamps to 0-1
        if np.equal(x_values,0).any():  # over-ride zero values at startup that will skew plot
            x_values = np.linspace(0,1,x_values.size)
        pts_x = x_values - np.min(x_values)
        pts_x = pts_x / np.max(pts_x)
        
        # Calc y-axis range, to nearest multiple
        y_max = np.ceil(np.max(y_values)/self.fps_scale_minor_unit) * self.fps_scale_minor_unit
        y_min = np.floor(np.min(y_values)/self.fps_scale_minor_unit) * self.fps_scale_minor_unit
                
        text_w2, text_h2 = cv2.getTextSize('%d' % y_max, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        text_w1, text_h1 = cv2.getTextSize('%d' % y_min, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        
        # Scale fps_values to 0-1 in y_range
        pts_y = y_values - y_min
        pts_y = pts_y / np.max((1,y_max-y_min))
        
        pts = np.stack([pts_x, pts_y], axis=1)
        
        # Scale to fill region, and leave space on the L for the axis text
        pts = (pts * np.array([[self.w-text_w2, self.plot_h]])).astype(np.int32)     
        pts[:, 0] = pts[:, 0] + text_w2
        cv2.polylines(output_buffer,[pts], isClosed=False, color=(255,255,255,255*self.plot_alpha), thickness=self.line_width)     
        
        output_buffer = cv2.flip(output_buffer, 0)  # flip (so low is at bottom)
        
        # Axis
        cv2.line(output_buffer, (text_w2, 0), (text_w2, self.h), (255,255,255,255*self.plot_alpha), self.axes_lineThickness)
        draw_str(output_buffer, (0, text_h2+2), '%d' % y_max, 'L')
        draw_str(output_buffer, (text_w2, self.plot_h-2), '%d' % y_min, 'R')
        
        return output_buffer
    
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n      

def conv_smooth(y, box_pts):
    # 1-d moving av convolutional smoothing
    # https://stackoverflow.com/a/26337730/5859283
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


class SubImage(Widget):
    # Adds a subimage (picture-in-picture) to the image
    w = 200
    h = 200
    update_period = .1
    rects_buffer = []
    num_face_detect_max = 1
    target_found = False
    img_hold_max = 10  # update cycles to hold the subImage if target not found, before it starts to disappear
    img_fade_point = 5  # value of img_hold_counter at which the image starts fading
    img_hold_counter = img_hold_max    
    face_cascade_filename = "haarcascade_frontalface_default.xml"  # "haarcascade_frontalface_alt.xml"
    eye_cascade_filename = "haarcascade_eye.xml"

    def __init__(self, interface, position='TL', size=None):
        super().__init__(interface, position, size)

        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_filename)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_filename)
        
        self.detector = FaceTracker()

    def update(self):
        
        if self.interface.last_frame is not None:
            img = np.copy(self.interface.last_frame)
            
            if len(img.shape)>2 and img.shape[-1]==4:
                img = img[...,:3]  # remove alpha
                
            # Run the detector, and store the output for later use
            rect_results = self.detector.detect(img)
            
            # Picture-in-picture
            if len(rect_results)>0:  # target has been found in main image
                # Found a face - > update output
                (x1, y1, x2, y2) = rect_results[0]  # use main region for the sub-image
                sub_img = np.copy(img[y1:y2, x1:x2])
                
                output_buffer = np.zeros((self.h, self.w, 4)).astype(np.uint8)
                output_buffer[...,:3] = cv2.resize(sub_img, (self.h, self.w))
                output_buffer[...,3] = 255 * np.ones((self.h, self.w)).astype(np.uint8)
            
                self.output = output_buffer
                self.rects_buffer = rect_results
                
                if not self.target_found: # target has just been refound - reset flags and counter
                    self.target_found = True
                    self.img_hold_counter = self.img_hold_max
                
            else:  # target has not been found, prepare to or actually get rid of the subImage
                self.target_found = False
                if self.img_hold_counter>0:
                    self.img_hold_counter -= 1
                if self.img_hold_counter == 0:
                    self.output = None  # reset_output()  # blank out PIP
                    self.rects_buffer = []
                elif (self.img_hold_counter < self.img_fade_point) and (self.output is not None): # start fading the image
                    self.output[...,3] = self.output[...,3] * 0.5
    
    def draw_rects(self, this_img, rects, color):
        for x1, y1, x2, y2 in rects[:self.num_face_detect_max]:
            cv2.rectangle(this_img, (x1, y1), (x2, y2), color, 2)
            
    def direct_write(self, this_img):
        # called for every screen refresh - use sparingly!
        self.draw_rects(this_img, self.rects_buffer, (0, 255, 0, 255))


class FaceTracker:
    face_cascade_filename = "haarcascade_frontalface_default.xml"  # "haarcascade_frontalface_alt.xml"
    eye_cascade_filename = "haarcascade_eye.xml"
    require_eyes = True
    num_face_detect_max = 1  # Hardcoded for now
    face_centroid = None
    face_rect = None
    face_centroid_delta_threshold = 25  # Update the face position if it has moved more than this many pixels (Euclidean)
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_filename)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_filename)

    
    def detect(self, img):
        
        # Merge channels           
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
            
        face_rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Convert coordinates and limit number of rectangles
        face_rects = self.convert_coord_sys(face_rects[:self.num_face_detect_max])
        
        if len(face_rects) == 0:
            return []
            
        if self.require_eyes:
            eyes_found = np.zeros(face_rects.shape[0])
            for i, (x1, y1, x2, y2) in enumerate(face_rects):
                sub_img = gray[y1:y2, x1:x2]
                
                eye_rects = self.eye_cascade.detectMultiScale(sub_img, scaleFactor=1.5, minNeighbors=2, minSize=(15, 15),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
                if len(eye_rects)>0:
                    eyes_found[i] = 1
                    
                    # Draw box around eyes
#                    vis_roi = img[y1:y2, x1:x2]
#                    self.draw_rects(vis_roi, self.convert_coord_sys(eye_rects), (0, 255, 0, 255))
                    
        if not np.any(eyes_found):
            return []
        else:
            face_rects = face_rects[np.nonzero(eyes_found)[0]] 
        
        # Decide whether to update the face position
        face_centroid = (face_rects[:,:2] + face_rects[:,2:]) / 2
        
        if self.face_centroid is not None:
            delta = np.linalg.norm(self.face_centroid - face_centroid)

            if delta > self.face_centroid_delta_threshold:
                # Update face position
                self.face_centroid = face_centroid
                self.face_rect = face_rects
                return face_rects
            else:
                # Return old position, no update required
                return self.face_rect
            
        else:
            # No old data so use new position
            self.face_centroid = face_centroid
            self.face_rect = face_rects
            return face_rects
    
    def draw_rects(self, this_img, rects, color):
        for x1, y1, x2, y2 in rects[:self.num_face_detect_max]:
            cv2.rectangle(this_img, (x1, y1), (x2, y2), color, 2)
            
    def convert_coord_sys(self, rects):
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects
  
    