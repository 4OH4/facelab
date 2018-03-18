# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:57:15 2018

@author: Rupert.Thomas

Front-end interface for facelab
Real-time system for image recognition/processing of faces
"""

import numpy as np
import cv2 as cv2

# local modules
from WebcamVideoStream import WebcamVideoStream
from Widget import Timer, SubImage, HistogramPlot, FPS_plot

class Interface:
        
    last_frame = None
    frame_count = 0
    
    def __init__(self):
        
        # Setup camera
        cam_id = 0
        self.mirror = False
        self.cam = WebcamVideoStream(src=cam_id).start()
        
        self.gui_w = self.cam.w
        self.gui_h = self.cam.h
        
        # Create processing modules list
        self.module_data = {}
        timer = Timer(self, position='TR')
        sub_image = SubImage(self, position='TL')
        fps_plot = FPS_plot(self, position='BR')
        hist_plot = HistogramPlot(self, position='BL')
       
        self.modules = [timer, hist_plot, fps_plot, sub_image]
        

    def run(self):
        
        # debug        
#        img = cv2.imread('test_image.jpg')
#        img = cv2.resize(img, (960,540)).astype(np.uint8)
        
        for module in self.modules:
            module.update()
            module.thread.start()
        
        while True:
            img = self.cam.read()
            
            # debug
#            img = np.random.choice([10,100,166], (self.gui_h, self.gui_w, 3)).astype(np.uint8)  #, dtype=np.uint8

            if img is not None:
                self.last_frame = img  # .copy()
                self.frame_count += 1
                
                output_image = img.copy()
                
                for module in self.modules:
                    # Add each overlay to the output image
                    if module.output is not None:
                        resized_overlay = cv2.resize(module.output.copy(), (module.w, module.h)).astype(np.uint8)
                    
                        output_image[module.y:module.y+module.h, module.x:module.x+module.w] = \
                                blend_transparent( output_image[module.y:module.y+module.h, module.x:module.x+module.w], 
                                          resized_overlay)
                            
                    # Allow direct writes to the image as well
                    module.direct_write(output_image)
                        
                cv2.imshow('facelab', output_image)
                
            else:
                print("img is none")
            
            key = cv2.waitKey(1)
            if key == 27:
                break

        self.clean_up()
            
            
    def clean_up(self):
        # Get rid of any remaining components and shut down the threads
        cv2.destroyAllWindows()        
        for module in self.modules:
            module.running = False
            module.thread.join()
            

def blend_transparent(face_img, overlay_t_img):
    # Overlay images using the alpha from the top one
    
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))



#%%

disp = Interface()
disp.run()

