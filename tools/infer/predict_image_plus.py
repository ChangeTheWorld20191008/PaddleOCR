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
from ppocr.utils.utility import get_image_file_list

from tools.infer.my_utils import ObjectDetector
from tools.infer.my_utils import TextDetector
from tools.infer.my_utils import draw_text_det_res
from tools.infer.my_utils import text_and_car_iou
from tools.infer.my_utils import aspect_ratio_filter

logger = get_logger()

if __name__ == "__main__":
    args = utility.parse_args()

    model_path = '/home/zhuhao/myModel/person_and_car/standard'
    label_file = '/home/zhuhao/myModel/person_and_car/standard/label.names'
    object_name = 'car'
    object_detector = ObjectDetector(model_path, label_file)

    text_detector = TextDetector(args)
    video_file = args.video_file
    video_save_file = args.video_save_file

    iou_thresh = 0.8
    lp_aspect_ratio = [1.1, 3.9]

    draw_img_save = args.draw_img_save_dir
    image_file_list = get_image_file_list(args.image_dir)

    imagecount = 0
    for i, image_file in enumerate(image_file_list):
        image = cv2.imread(image_file)

        logger.info(f"The {i+1} image is detecting.")

        # do car detector
        car_boxes, car_scores = object_detector.detect(image, object_name)

        # do text detector
        text_boxes, _, text_scores = text_detector(image)

        # do license plate size filter
        size_filter_boxes = []
        size_filter_scores = []
        for text_box, score in zip(text_boxes, text_scores):
            # logger.info(f"[TMP]: text_box is {text_box} and score is {score}")
            if aspect_ratio_filter(text_box, lp_aspect_ratio):
                size_filter_boxes.append(text_box)
                size_filter_scores.append(score)

        # do iou
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

            max_iou = 0.0
            for car_box in car_boxes:
                iou_score = text_and_car_iou(car_box, text_box_sim)
                if iou_score > iou_thresh:
                    dt_boxes.append(text_box)
                    scores.append(score)
                    break

        res_im = draw_text_det_res(image, dt_boxes, scores)

        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, res_im)

    object_detector.close()
