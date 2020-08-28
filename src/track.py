import numpy as np
import cv2
import os

class Detection:
    def __init__(self, box, state, frame):
        self.box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        self.state = state
        self.frame = frame
        self.pt1 = (box[0],box[1])
        self.pt2 = (box[2],box[3])


class Track:
    def __init__(self, box, state, frame, frame_n, t_id, violation_dir, out_fps=3, state_transition_th=2):
        self.bboxes = []
        self.state_list = []
        self.frames = []
        
        self.state = 0
        self.t_id = t_id
        
        self.state_transition_th = state_transition_th
        self.transitions = 0
        
        self.images = []
        self.violation = False
        
        self.violation_dir = violation_dir
        self.out_fps = out_fps
        
        self.miss_count = 0
        self.update(box, state, frame_n, frame)
            
    def update(self, box, state, frame_n, frame):
        
        self.miss_count = 0
        self.bboxes.append(box) 
        self.state_list.append(state) 
        self.frames.append(frame_n)
        
        self.position = box
        self.update_state(state)
        self.store_frame(frame)
        
    def miss_track(self, frame_n):
        
        self.bboxes.append(self.bboxes[-1]) 
        self.state_list.append(self.state_list[-1]) 
        self.frames.append(frame_n)
        
        self.miss_count += 1
        
    def update_state(self, state):
        
        if self.state != state:
            self.transitions += 1
            if self.transitions >= self.state_transition_th:
                self.change_state(state)
        else:
            self.transitions = 0
                        
    def change_state(self, state):
        if self.state == 0:
            self.violation = True
        else:
            self.violation = False
            self.save_violation_clip()
            
        self.state = state
        self.transitions = 0
                        
    def store_frame(self, frame):
        if self.violation:
            self.images.append(frame)
        
    def save_violation_clip(self):
        vid_file = os.path.join(self.violation_dir, str(self.t_id)+'.avi')
        if len(self.images) > 0:
            out = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc('M','J','P','G'), 
                                  self.out_fps, (self.images[0].shape[1],self.images[0].shape[0] ))
            for img in self.images:
                out.write(img) 
        
    def frame_count(self):
        return len(self.frames)