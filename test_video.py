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

# MinIO
# minio_address = '10.248.243.110:9000'
# bucket_name = 'ai-images'
# client = Minio(
#     minio_address,
#     access_key='minioadmin',
#     secret_key='Vcc_AI@2022',
#     secure=False
# )
# # Setup Kafka
# face_recognition_topic = 'faceRecognition'
# face_alert_topic = 'faceUnknownAlert'
# kafka_broker = '10.248.243.110:39092'
# producer = kafka.KafkaProducer(bootstrap_servers=kafka_broker)

# # Postgres
# conn = psycopg2.connect(host='10.248.243.110', database='vcc_ai_events', user='postgres', password='Vcc_postgres@2022', port=5432)
# conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
# cur = conn.cursor()

# models = load_model()
class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rtsp = "rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1"
        # self.rtsp = '/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/videos_input/GOT_actor.mp4'
        # self.coordinates = [[950, 640], [1000, 740], [1700, 640], [1670, 540]]
        self.model_retinaface = load_model.retinaface
        self.model_arcface = load_model.arcface
    
    def run(self):
        frame_count = 0
        face_all = []
        check_appear = []
        cap = cv2.VideoCapture(self.rtsp)
        # box = self.coordinates
        # box_int32 = np.array(box, np.int32)
        count_unknown = 0
        nothing = 0
        while True:
            
            try:
                det = []
                counter = collections.Counter(check_appear) # Counter({'a': 4, 'c': 2, 'b': 1})
                a = time.time()
                timer = cv2.getTickCount()
                ret, frame = cap.read()
                # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Inference
                detect_tick = time.time()
                result_boxes, result_scores, result_landmark = self.model_retinaface.infer(frame_rgb)
                # if len(result_boxes) == 0 and nothing <= 50:
                #     nothing += 1
                # elif  nothing >= 50:
                #     check_appear.clear()
                #     nothing = 0
                # else: nothing = 0
                # print('nothing:',nothing)
                detect_tock = time.time()
                # print("Faces detection time: {}s".format(detect_tock-detect_tick))
                for i in range(len(result_boxes)):
                    bbox = np.array([result_boxes[i][0], result_boxes[i][1], result_boxes[i][2], result_boxes[i][3]])
                    landmarks = result_landmark[i]
                    landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                        landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                    landmark = landmark.reshape((2,5)).T
                    scores = result_scores[i]

                    nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

                    # Arcface_paddle
                    labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
                    
                    # if counter[labels[0]] <= 30:
                    #     check_appear.append(labels[0])
                    
                    # if labels[0] not in  face_all and labels[0] != 'unknown':
                        
                    #     face_all.append(labels[0])
                    # elif labels[0] == 'unknown' and count_unknown >= 10:
                    #     count_unknown += 1
                    #     print("SEND")

                    # print('face_all:',face_all)
                    # print('check_appear:',check_appear)
                    # print('c:',counter)
                    # Draw
                    plot_one_box(
                        result_boxes[i],
                        landmarks,
                        frame,
                        label="{}-{:.2f}".format(labels[0], scores))

                b = time.time() - a
                # print("Total time: {}s".format(b))
                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                print("FPS:", round(FPS))
                cv2.putText(frame, 'FPS: ' + str(int(FPS)), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 70), 2)
                frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
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