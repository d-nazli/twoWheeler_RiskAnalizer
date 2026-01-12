import cv2
import numpy as np

class DenseFlow:
    
    def __init__(self,pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
             
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.flow = None
        self.optical_flow_colormap = None
    def size_matching(self, frame1, frame2):
     
        
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        max_h = max(h1, h2)
        max_w = max(w1, w2)
    
        frame1_resized = cv2.resize(frame1, (max_w, max_h))
        frame2_resized = cv2.resize(frame2, (max_w, max_h))
        
        return frame1_resized, frame2_resized  # Yeni boyutlandırılmış çerçeveleri döndür

    
    def compute_dens_flow(self,first_frame,second_frame):
        
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

        self.flow = cv2.calcOpticalFlowFarneback(
            first_frame_gray, second_frame_gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, self.flags
        )
        return self.flow
    
    def vis_optical_flow_colormap(self,first_frame):
        
        hsv = np.zeros_like(first_frame)
        hsv[..., 1] = 255  # Doygunluğu maksimum yap
        
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # Açıları renklendir
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Büyüklüğü normalize et
        
        self.optical_flow_colormap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return self.optical_flow_colormap
    
    





        
