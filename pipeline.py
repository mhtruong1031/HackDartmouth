from google import genai
from google.genai import types
from PIL import Image
from config import API_KEY

import mediapipe as mp
import numpy as np
import cv2
import pyttsx3

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

        self.gold_standard = {
            "Push-up": self.video_to_landmarks('push_up.mp4'),
            "Squat": self.video_to_landmarks('squat.mp4')
        }

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160) # speaking speed

    def run(self, video_path: str):
        mp_output           = self.video_to_landmarks(video_path, True)
        self.classification = self.identify_exercise('frame.jpg')
        standard            = self.gold_standard[self.classification]
        user_isolated_rep   = self.isolate_rep(mp_output, standard[0])

        self.vectorized_angle_data = self.error_calculation(user_isolated_rep, standard, self.classification)
        return self.generate_improvement()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    # returns list of frames with mp outputs
    def video_to_landmarks(self, video_path: str, save: bool = False):
        cap = cv2.VideoCapture(video_path)

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils

        first_frame = True
        frames = []

        # If saving, setup VideoWriter
        if save:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("output_overlay.mp4", fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if first_frame:
                cv2.imwrite('frame.jpg', frame)
                first_frame = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            frames.append(result)

            if save and result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

            if save:
                out.write(frame)

        cap.release()
        if save:
            out.release()
            print("Saved overlay video as output_overlay.mp4")

        return frames

    
    # frames - list of frames' results from Pose().process()
    # n_conescutive - int for how many consecutive frames to check for
    # Returns the frame with the range of the first isolated rep (int, int)
    def isolate_rep(self, frames: list, ref_pose, n_consecutive: int = 2) -> tuple[int, int]:
        start_i = 0
        mid_i   = 0
        end_i   = 0

        # Find rep start
        last_dist    = 1e-99
        max_ref_dist = 1e99
        streak       = 0

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
                mid_i = i
                break

            last_dist = dist

        last_dist    = 1e-99
        max_ref_dist = 1e99
        streak       = 0

        for i, curr_frame in enumerate(frames[mid_i:], start=mid_i):
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

        return frames[start_i: end_i]

    def identify_exercise(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="You can only respond with one of the following words: Push-up, Sit-up, Squat"),
            contents=[image, "What exercise is this?"]
        )
        return response.text.strip()
    
    # Returns the angle difference between the matched up frames
    def error_calculation(self, video_vectors, golden_vectors, classification):
        video_vectors_len = len(video_vectors)
        golden_vectors_len = len(golden_vectors)

        #lengthening or shortening effect 
        if video_vectors_len < (golden_vectors_len - 30):
            duration_condition = "fast"
        elif video_vectors_len > (golden_vectors_len + 30):
            duration_condition = "slow"
        else:
            duration_condition = "good speed"

        #matching up and subtracting frames
        video_angles = []
        golden_angles = []
        print(len(video_vectors))
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

    def generate_improvement(self):
        improvement = "You are doing " + str(self.classification) + "s! "
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction = f"You are an exercise coach helping with {self.classification}. The user's reps are at a {self.vectorized_angle_data.get('USER_EXERCISE_SPEED', 'unknown')} speed. Here is the user's data showing error in angle vs. a gold standard over time: {str(self.vectorized_angle_data)}. Keep it centered around specific joints that are high in error. Specifically mention the error numbers and interpretations. Limit to under 30 words."),
            contents=["Please analyze the user's form and provide improvement suggestions."]
        )
        return improvement + response.text.strip()

    # Returns the eclidiean distance between 2 frames
    # res_1/res_2 - result object from Pose().process()
    def __calculate_frame_euclidean_distance(self, res_1, res_2) -> float:
        # Remove non-existent points
        low_vis_filter = set(self.class_filters.get(self.classification))
        for i, ld in enumerate(res_1.pose_landmarks.landmark):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)
        for i, ld in enumerate(res_2.pose_landmarks.landmark):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)

        vec_1 = self.__vectorize_res(res_1, low_vis_filter)
        vec_2 = self.__vectorize_res(res_2, low_vis_filter)

        return np.linalg.norm(vec_1 - vec_2)
    
    def __vectorize_res(self, result, low_vis_filter: set = set()) -> np.array:
        vector = []
        for i, ld in enumerate(result.pose_landmarks.landmark):
            if i not in low_vis_filter:
                vector.extend([ld.x, ld.y, ld.z])

        return np.array(vector)

    def __anglize_res(self, result, low_vis_filter: set = set()) -> np.array:
        low_vis_filter = set(self.class_filters.get(self.classification))
        for i, ld in enumerate(result.pose_landmarks.landmark):
            if ld.visibility < 0.5:
                low_vis_filter.add(i)

        def landmark_to_np(landmark):
            return np.array([landmark.x, landmark.y, landmark.z])

        vector = []
        for i, data in enumerate(self.angle_nodes):
            if self.classification == 'squat' and (data[2] == 12 or data[2] == 15):
                continue

            # Pull coordinates
            p0 = landmark_to_np(result.pose_landmarks.landmark[data[0]])
            p1 = landmark_to_np(result.pose_landmarks.landmark[data[1]])
            p2 = landmark_to_np(result.pose_landmarks.landmark[data[2]])

            v_1 = p0 - p1
            v_2 = p2 - p1

            # Safe angle computation
            dot_product = np.dot(v_1, v_2)
            norms_product = np.linalg.norm(v_1) * np.linalg.norm(v_2)

            if norms_product == 0:  # Prevent division by zero
                angle = 0.0
            else:
                cosine_angle = np.clip(dot_product / norms_product, -1.0, 1.0)  # Clip for safety
                angle = np.arccos(cosine_angle)

            vector.append(angle)

        return np.array(vector)
    
