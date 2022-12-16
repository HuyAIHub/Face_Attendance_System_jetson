import time
from datetime import datetime
import numpy as np
import cv2
from shapely.geometry import Polygon
from utils import img_warped_preprocess, plot_one_box
import threading
from load_model import load_model
import threading
import kafka
from minio import Minio
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import collections
# Initialize some variables
from sort import Sort
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
color=(0,0,255)
count = 0
ids = []


class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rtsp = "/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/videos_input/GOT_actor.mp4"
        self.model_retinaface = load_model.retinaface
        self.model_arcface = load_model.arcface
    
    def run(self):
        cap = cv2.VideoCapture(self.rtsp)

        while True:
            
            try:
                det = []
                ret, frame = cap.read()
                # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inference
                result_boxes, result_scores, result_landmark = self.model_retinaface.infer(frame_rgb)
                print('=================================================')
                # print('result_boxes',result_boxes)
                # print('result_landmark:',result_landmark)
                predict = tracker.update(result_boxes)
                boxes_track = predict[:,:-1]
                boces_ids = predict[:,-1].astype(int)
                for i, (bbox,landmarks,id_, score) in enumerate(zip(boxes_track, result_landmark,boces_ids, result_scores)):
                    landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                        landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                    print('landmark before zip:',landmark)
                    landmark = landmark.reshape((2,5)).T
                    print('bbox zip:',bbox)
                    print('landmark zip:',landmark)
                    nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    cv2.imshow('check',nimg)
                    # Arcface_paddle
                    labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
                    plot_one_box(
                        result_boxes[i],
                        landmark,
                        frame,
                        label="{}-{:.2f}".format(labels[0], score),
                        id=id_)
                # for i in range(len(result_boxes)):
                #     bbox = np.array([result_boxes[i][0], result_boxes[i][1], result_boxes[i][2], result_boxes[i][3]])
                    
                #     landmarks = result_landmark[i]
                #     landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                #                         landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                   
                #     landmark = landmark.reshape((2,5)).T
                #     scores = result_scores[i]
                #     bbox = np.array([result_boxes[i][0], result_boxes[i][1], result_boxes[i][2], result_boxes[i][3]])
                #     landmarks = result_landmark[i]
                #     landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                #                         landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                #     print('landmark=====before:',landmark)
                #     landmark = landmark.reshape((2,5)).T
                #     scores = result_scores[i]    
                #     print('bbox=====:',bbox)
                #     print('landmark=====:',landmark)
                #     nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                #     nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                #     cv2.imshow('check2',nimg)
                #     # Arcface_paddle
                #     labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
                #     # Draw
                #     for id in boces_ids:
                #         plot_one_box(
                #             result_boxes[i],
                #             # box,
                #             landmarks,
                #             frame,
                #             label="{}-{:.2f}-{}".format(labels[0], scores,id))

                frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                cv2.imshow("vid_out", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except Exception as error:
                print("Error:",error)
                self.model_retinaface.destroy()
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(1)

if __name__ == '__main__':
    runModel = RunModel()
    runModel.start()
# python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# opencv-python==4.2.0.32