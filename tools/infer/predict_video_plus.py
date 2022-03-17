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

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger

from tools.infer.motion_detector import MotionDetector
from tools.infer.my_utils import ObjectDetector
from tools.infer.my_utils import TextDetector
from tools.infer.my_utils import draw_text_det_res
from tools.infer.my_utils import main_and_inter_iou
from tools.infer.my_utils import aspect_ratio_filter

logger = get_logger()


if __name__ == "__main__":
    args = utility.parse_args()

    model_path = '/home/zhuhao/myModel/person_and_car/standard'
    label_file = '/home/zhuhao/myModel/person_and_car/standard/label.names'
    object_name = 'car'
    object_detector = ObjectDetector(model_path, label_file)

    text_detector = TextDetector(args)

    motion_detector = MotionDetector(bg_history=10)

    video_file = args.video_file
    video_save_file = args.video_save_file

    # license plate size filter param
    lp_aspect_ratio = [1.1, 4.9]
    # motion region filter param
    motion_iou_thresh = 0.5
    # car region filter param
    text_iou_thresh = 0.8
    car_iou_thresh = 0.15

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f'[Error] opening input video: {video_file}')
        sys.exit(0)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_ptr = cv2.VideoWriter(
        video_save_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # do car detector
        car_boxes, _ = object_detector.detect(frame, object_name)

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
        dt_boxes = []
        scores = []
        for text_box, score in zip(size_filter_boxes, size_filter_scores):
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
                text_iou, car_iou = main_and_inter_iou(
                    text_box_sim, car_box)
                if text_iou >= text_iou_thresh and car_iou <= car_iou_thresh:
                    dt_boxes.append(text_box)
                    scores.append(score)
                    break

        res_im = draw_text_det_res(frame, dt_boxes, scores)

        out_ptr.write(res_im)

        frame_count += 1
        logger.info(f"{frame_count} is detecting. score is {scores}")

    if out_ptr:
        out_ptr.release()

    if cap:
        cap.release()

    object_detector.close()
