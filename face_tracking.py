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
import collections
# Initialize some variables
from sort import Sort
face_check = []
name_check = []
id_check = []
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        # self.rtsp = "/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/data_test/video3.avi"
        # self.rtsp = '/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/videos_input/GOT_actor.mp4'
        self.rtsp = 'rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1'
        self.model_retinaface = load_model.retinaface
        self.model_arcface = load_model.arcface
    
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)

        while True:
            timer = cv2.getTickCount()
            try:
                ret, frame = cap.read()
                # frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inference
                result_boxes, result_scores, result_landmark = self.model_retinaface.infer(frame_rgb)
                print('=================================================')
                predict=tracker.update(result_boxes)
                print('predict',predict)
                boxes_track = predict[:,:-1]
                boces_ids = predict[:,-1].astype(int)
                
                for i, (bbox,landmarks,id_, score) in enumerate(zip(boxes_track, result_landmark,boces_ids, result_scores)):
                    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
                    landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                        landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                    landmark = landmark.reshape((2,5)).T
                    # Align face
                    nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    cv2.imshow('check',nimg)
                    # Arcface_paddle
                    labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
                    # Draw face
                    if id_ >= 6:
                        tracker.reset_count()
                    plot_one_box(
                        result_boxes[i],
                        landmark,
                        frame,
                        label="{}-{:.2f}".format(labels[0], score),
                        id=id_)
                    # SEND
                    print('labels:',labels)
                    print('score:',score)
                    if len(labels) != 0 and score > 0.98 and id_ not in id_check :
                        print('========================SEND=====================')
                        cv2.imwrite('/home/aitraining/workspace/huydq46/Face_Attendance_System/output/'+labels[0]+'_'+current_time+'.jpg',frame)
                    id_check.append(id_)
                frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # print("FPS:", round(FPS))
                cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
                cv2.imshow("vid_out", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_count += 1
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
