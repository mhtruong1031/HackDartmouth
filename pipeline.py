from google import genai
from google.genai import types
from PIL import Image
from config import API_KEY

import mediapipe as mp
import numpy as np
import cv2

class Pipeline:
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY)

        self.angle_nodes = [
            (16, 14, 12),
            (11, 13, 15),
            (12, 24, 26),
            (11, 23, 25),
            (24, 26, 28),
            (23, 25, 27)
        ]

        self.class_filters = {
            "Push-up": list(range(11)) + list(range(16, 23)) + list(range(29, 33)),
            "Sit-up": list(range(11)) + list(range(16, 23)) + list(range(29, 33)),
            "Squat": list(range(29, 33))
        }

    def run(self, video_path: str):
        mp_output           = self.video_to_landmarks(video_path)
        self.classification = self.identify_exercise('balls.jpg')

        user_isolated_rep   = self.isolate_rep(mp_output)
        gold_standard       = self.video_to_landmarks() # TO-DO: Gold standard videos

        vectorized_angle_difference_data = self.error_calculation(user_isolated_rep, gold_standard, self.classification)

    
    # returns list of frames with mp outputs
    def video_to_landmarks(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        first_frame = True
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if first_frame:
                cv2.imwrite('balls.jpg', frame)
                first_frame = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(frame_rgb)
            frames.append(result)
        
        return frames
    
    # frames - list of frames' results from Pose().process()
    # n_conescutive - int for how many consecutive frames to check for
    # Returns the frame with the range of the first isolated rep (int, int)
    def isolate_rep(self, frames: list, ref_pose, n_consecutive: int = 2) -> tuple[int, int]:
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
                streak += 1
            else:
                streak = 0

            if streak >= n_consecutive:
                break

            last_dist = dist

        max_ref_dist  = 1e99
        for i, curr_frame in enumerate(frames[start_i+1:]):
            if i == 0:
                continue
            
            ref_dist = self.__calculate_frame_euclidean_distance(curr_frame, ref_pose)
            if ref_dist < max_ref_dist:
                max_ref_dist = ref_dist
                end_i        = i
            
            # Check for increase
            dist = self.__calculate_frame_euclidean_distance(curr_frame, frames[0])
            if dist > last_dist:
                streak += 1
            else:
                streak = 0

            if streak >= n_consecutive:
                break

            last_dist = dist

        return frames[start_i, end_i]

    def identify_exercise(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="You can only respond with one of the following words: Push-up, Sit-up, Squat"),
            contents=[image, "What exercise is this?"]
        )
        return response.text
    
    # Returns the angle difference between the matched up frames
    def error_calculation(self, video_vectors, golden_vectors, classification):
        video_vectors_len = len(video_vectors)
        golden_vectors_len = len(golden_vectors)

        #lengthening or shortening effect 
        if video_vectors_len < (golden_vectors_len-30):
            duration_condition = "fast"
        elif video_vectors_len > (golden_vectors_len+ 30):
            duration_condition = "slow"
        else:
            duration_condition = "good speed"

        #matching up and subtracting frames
        video_angles = []
        golden_angles = []
        for i, curr_frame in enumerate(video_vectors):
            percent_location_video = i/video_vectors_len

            video_angles_v = self.__anglize_res(curr_frame)
            golden_angles_v = self.__anglize_res(golden_vectors[int(percent_location_video * golden_vectors_len)])


            video_angles.append(video_angles_v)
            golden_angles.append(golden_angles_v)

            #squat doesn't need arms
            if classification == "Squat":
                indices = [2, 3, 4, 5]
            else:
                indices = [0, 1, 2, 3, 4, 5]

            angle_names = ['RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_HIP_HINGE', 'LEFT_HIP_HINGE', 'RIGHT_KNEE', 'LEFT_KNEE']

            #putting the difference into diff
            diff = {}

            # Initialize the dictionary
            for i in indices:
                landmark_name = angle_names[i]
                diff[landmark_name] = []
            diff['USER_EXERCISE_SPEED'] = duration_condition

            # Fill the dictionary
            for v_angle, g_angle in zip(video_angles, golden_angles):
                for i in indices:
                    landmark_name = angle_names[i]
                    diff[landmark_name].append(v_angle[i] - g_angle[i])
            
            return diff

    # Returns the eclidiean distance between 2 frames
    # res_1/res_2 - result object from Pose().process()
    def __calculate_frame_euclidean_distance(self, res_1, res_2) -> float:
        # Remove non-existent points
        low_vis_filter = set() + set(self.class_filters.get(self.classification))
        for i, ld in enumerate(res_1.pose_landmarks):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)
        for i, ld in enumerate(res_2.pose_landmarks):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)

        vec_1 = self.__vectorize_res(res_1, low_vis_filter)
        vec_2 = self.__vectorize_res(res_2, low_vis_filter)

        return np.linalg.norm(vec_1 - vec_2)
    
    def __vectorize_res(self, result, low_vis_filter: set = set()) -> np.array:
        vector = []
        for i, ld in enumerate(result.pose_landmarks):
            if i not in low_vis_filter:
                vector.extend([ld.x, ld.y, ld.z])

        return np.array(vector)

    def __anglize_res(self, result, low_vis_filter: set = set()) -> np.array:
        vector = []
        for i, data in self.angle_nodes:
            if self.classification == 'squat' and (data[2] == 12 or data[2] == 15):
                continue

            v_1 = result.pose_landmarks.landmark[data[0]] - result.pose_landmarks.landmark[data[1]]
            v_2 = result.pose_landmarks.landmark[data[2]] - result.pose_landmarks.landmark[data[1]]

            angle = np.arccos(np.dot(v_1, v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2)))
            vector.append(angle)

        return np.array(vector)
    
