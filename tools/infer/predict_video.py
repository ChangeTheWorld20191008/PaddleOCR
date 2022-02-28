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
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow.python.saved_model import tag_constants

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
logger = get_logger()


class ObjectDetector:
    def __init__(self, model_path='./model', label_file='./model/label.names',
                 num_classes=2, score_threshold=0.5, image_sz=(416, 416, 3)):
        self._model_path = model_path
        self._label_file = label_file
        self._num_classes = num_classes
        self._score_threshold = score_threshold
        self._image_sz = image_sz[0:2]

        self._config = ConfigProto()
        self._config.gpu_options.allow_growth = True

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._sess = tf.Session(config=self._config)

            tf.saved_model.load(
                self._sess, [tag_constants.SERVING], self._model_path)

            self._image_tensor = self._sess.graph.get_tensor_by_name(
                'serving_default_input_1:0')
            self._output_tensor = self._sess.graph.get_tensor_by_name(
                'StatefulPartitionedCall:0')

            self._boxes = tf.placeholder(
                tf.float32, shape=(None, None, None, 4))
            self._scores = tf.placeholder(
                tf.float32, shape=(None, None, self._num_classes))

            self._boxes_predi, self._scores_predi, self._classes_predi,\
                self._valid_detections_predi = \
                tf.image.combined_non_max_suppression(
                    boxes=self._boxes, scores=self._scores,
                    max_output_size_per_class=50, max_total_size=50,
                    iou_threshold=0.45, score_threshold=self._score_threshold)

            self._label_map = self._load_labelmap(self._label_file)

    def _load_labelmap(self, label_file):
        category_index = {}
        index = 1

        for line in open(label_file):
            category_index[index] = line.rstrip("\n")
            index += 1

        return category_index

    def detect(self, image, object_name):
        image_h, image_w, _ = image.shape
        ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(ori_image, self._image_sz)
        det_image = image_data / 255.
        image_np_expanded = np.expand_dims(det_image, axis=0)
        image_np_expanded = np.asarray(image_np_expanded).astype(np.float32)

        pred_bbox = self._sess.run(
            self._output_tensor,
            feed_dict={self._image_tensor: image_np_expanded})

        boxes_pred, scores_pred, classes_pred, valid_detections_pred = \
            self._sess.run(
                [self._boxes_predi, self._scores_predi, self._classes_predi,
                 self._valid_detections_predi],
                feed_dict={
                    self._boxes: np.reshape(
                        pred_bbox[:, :, 0:4],
                        (pred_bbox[:, :, 0:4].shape[0], -1, 1, 4)),
                    self._scores: pred_bbox[:, :, 4:]})

        boxes = boxes_pred[0][:valid_detections_pred[0]]
        scores = scores_pred[0][:valid_detections_pred[0]]
        classes = classes_pred[0][:valid_detections_pred[0]] + 1
        labels = [self._label_map[classes_id] for classes_id in classes]

        car_boxes = []
        car_scores = []
        for box, score, label in zip(boxes, scores, labels):
            if label == object_name:
                car_boxes.append(
                    [int(box[1] * image_w), int(box[0] * image_h),
                     int(box[3] * image_w), int(box[2] * image_h)])
                car_scores.append(score)

        return car_boxes, car_scores

    def close(self):
        self._sess.close()
        self._sess = None


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.use_onnx = args.use_onnx
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'det', logger)

        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        """
        reference from:
         https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['maps'] = outputs[0]

        # self.predictor.try_shrink_memory()
        post_result, score_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        scores = score_result[0]['score']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et = time.time()
        return dt_boxes, et - st, scores


def draw_text_det_res(img, dt_boxes, scores):
    for box, score in zip(dt_boxes, scores):
        box = np.array(box).astype(np.int32).reshape(-1, 2)

        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        left = box[0][0]
        top = box[0][1]
        cv2.putText(
            img, f"{score:.2f}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 2)
    return img


def text_and_car_iou(car_box, text_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(car_box[0], text_box[0])
    yA = max(car_box[1], text_box[1])
    xB = min(car_box[2], text_box[2])
    yB = min(car_box[3], text_box[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if inter_area == 0:
        return 0, 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    car_box_area = abs((car_box[2] - car_box[0]) * (car_box[3] - car_box[1]))
    text_box_area = abs(
        (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    text_iou = inter_area / float(text_box_area)
    iou = inter_area / float(car_box_area + text_box_area - inter_area)

    # return the intersection over union value
    return text_iou, iou


if __name__ == "__main__":
    args = utility.parse_args()

    model_path = '/home/zhuhao/myModel/person_and_car/standard'
    label_file = '/home/zhuhao/myModel/person_and_car/standard/label.names'
    object_name = 'car'
    object_detector = ObjectDetector(model_path, label_file)

    text_detector = TextDetector(args)
    video_file = args.video_file
    video_save_file = args.video_save_file

    text_iou_thresh = 0.8
    iou_thresh = 0.15

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
        car_boxes, car_scores = object_detector.detect(frame, object_name)

        # do text detector
        text_boxes, _, text_scores = text_detector(frame)

        # do iou
        dt_boxes = []
        scores = []
        for text_box, score in zip(text_boxes, text_scores):
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

            for car_box in car_boxes:
                text_iou, iou = text_and_car_iou(car_box, text_box_sim)
                if text_iou >= text_iou_thresh and iou <= iou_thresh:
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
