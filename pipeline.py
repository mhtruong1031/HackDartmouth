from google import genai
from google.genai import types
from PIL import Image
from config import API_KEY

import mediapipe as mp
import numpy as np
import cv2

class Pipeline:
    def __init__(self):
        pass

    def run(self, video_path: str):
        mp_output          = self.video_to_landmarks(video_path)
        classification     = self.identify_exercise()
        filtered_mp_output = self.filter_mp(mp_output)
        class_cut_output   = self.filter_class(filtered_mp_output, classification)
        isolated_rep       = self.__isolate_rep(class_cut_output)
    
    # returns list of frames with mp outputs
    def video_to_landmarks(video_path: str):
        cap = cv2.VideoCapture(video_path)

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        frames       = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(frame_rgb)
            frame.append(result)
        
        return frames
    
    # frames - list of frames' results from Pose().process()
    # n_conescutive - int for how many consecutive frames to check for
    # Returns the frame range of the first isolated rep (int, int)
    def isolate_rep(self, frames: list, n_consecutive: int, ref_pose) -> tuple[int, int]:
        start_i = 0
        end_i   = 0

        # Find rep start
        last_dist    = 1e-99
        max_ref_dist = 1e99
        for i, curr_frame in enumerate(frames):
            if i == 0:
                continue
            
            ref_dist = self.__calculate_frame_euclidean_distance(curr_frame, ref_pose)
            if ref_dist < max_ref_dist:
                max_ref_dist = ref_dist
                start_i      = i
            
            # Check for decrease
            dist = self.__calculate_frame_euclidean_distance(curr_frame, frames[0])
            if dist < last_dist: 
                break
            last_dist = dist

        max_ref_dist  = 1e99
        for i, curr_frame in enumerate(frames[start_i+1:]):
            if curr_frame == frames[0]:
                continue
            
            ref_dist = self.__calculate_frame_euclidean_distance(curr_frame, ref_pose)
            if ref_dist < max_ref_dist:
                max_ref_dist = ref_dist
                end_i        = i
            
            # Check for increase
            dist = self.__calculate_frame_euclidean_distance(curr_frame, frames[0])
            if dist > last_dist: 
                break
            last_dist = dist

        return (start_i, end_i)

    # Returns the eclidiean distance between 2 frames
    # res_1/res_2 - result object from Pose().process()
    def __calculate_frame_euclidean_distance(self, res_1, res_2) -> float:
        # Remove non-existent points
        low_vis_filter = set()
        for i, ld in enumerate(res_1.pose_landmarks):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)
        for i, ld in enumerate(res_2.pose_landmarks):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)

        vec_1 = self.__vectorize_res(res_1, low_vis_filter)
        vec_2 = self.__vectorize_res(res_2, low_vis_filter)

        return np.linalg.norm(vec_1 - vec_2)
    
    def __vectorize_res(result, low_vis_filter: set = set()) -> np.array:
        vector = []
        for i, ld in enumerate(result.pose_landmarks):
            if i not in low_vis_filter:
                vector.extend([ld.x, ld.y, ld.z])

        return np.array(vector)
