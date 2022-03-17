# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2

from tools.infer.motion_detector import MotionDetector
from tools.infer.my_utils import ObjectDetector
from tools.infer.my_utils import TextDetector
from tools.infer.my_utils import main_and_inter_iou
from tools.infer.my_utils import aspect_ratio_filter
import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
logger = get_logger()


if __name__ == "__main__":
    args = utility.parse_args()

    model_path = '/home/zhuhao/myModel/person_and_car/standard'
    label_file = '/home/zhuhao/myModel/person_and_car/standard/label.names'
    object_name = 'car'

    videos_path = '/home/zhuhao/video/car/test'
    snapshot_path = '/home/zhuhao/video/car/snapshot_plus'

    # license plate size filter param
    lp_aspect_ratio = [1.1, 3.9]
    # motion region filter param
    motion_iou_thresh = 0.5
    text_iou_thresh = 0.8
    iou_thresh = 0.15

    object_detector = ObjectDetector(model_path, label_file)
    text_detector = TextDetector(args)

    videos_list = os.listdir(videos_path)
    for video_file in videos_list:
        print(f"[INFO]: begin run {video_file} ============")
        motion_detector = MotionDetector()
        saved_car_sanpshot = False
        max_score = 0.0

        video_name = video_file.split('.')[0].split('/')[-1]
        cap = cv2.VideoCapture(f"{videos_path}/{video_file}")
        if not cap.isOpened():
            print(f'[Error] opening input video: {video_file}')
            sys.exit(0)

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # do car detector
            car_boxes, _ = object_detector.detect(frame, object_name)
            if not saved_car_sanpshot and len(car_boxes) != 0:
                cv2.imwrite(
                    os.path.join(snapshot_path, f"{video_name}_car.jpg"),
                    frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                saved_car_sanpshot = True

            # do text detector
            text_boxes, _, text_scores = text_detector(frame)

            # do motion detector
            motion_boxes = motion_detector.detect(frame)

            # do motion region filter
            motion_filter_boxes = []
            for car_box in car_boxes:
                for motion_box in motion_boxes:
                    motion_iou, _ = main_and_inter_iou(car_box, motion_box)
                    if motion_iou >= motion_iou_thresh:
                        motion_filter_boxes.append(car_box)
                        break

            # do license plate size filter
            size_filter_boxes = []
            size_filter_scores = []
            for text_box, score in zip(text_boxes, text_scores):
                if aspect_ratio_filter(text_box, lp_aspect_ratio):
                    size_filter_boxes.append(text_box)
                    size_filter_scores.append(score)

            # do license plate and car filter
            for text_box, score in zip(size_filter_boxes, size_filter_scores):
                if score > max_score:
                    xmin, ymin, xmax, ymax = 1000000, 1000000, 0, 0
                    for t_box in text_box:
                        if t_box[0] < xmin:
                            xmin = t_box[0]

                        if t_box[1] < ymin:
                            ymin = t_box[1]

                        if t_box[0] > xmax:
                            xmax = t_box[0]

                        if t_box[1] > ymax:
                            ymax = t_box[1]

                    text_box_sim = [xmin, ymin, xmax, ymax]

                    for car_box in motion_filter_boxes:
                        text_iou, iou = main_and_inter_iou(
                            text_box_sim, car_box)
                        if text_iou >= text_iou_thresh and iou <= iou_thresh:
                            cv2.imwrite(
                                os.path.join(
                                    snapshot_path,
                                    f"{video_name}_license_plate.jpg"),
                                frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            max_score = score
                            break

            frame_count += 1
            logger.info(
                f"{frame_count} is detecting. max_score is {max_score}")

        if cap:
            cap.release()

    object_detector.close()
