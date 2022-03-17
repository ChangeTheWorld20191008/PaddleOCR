import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow.python.saved_model import tag_constants

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.postprocess import build_post_process
from ppocr.data import create_operators
from ppocr.data import transform

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


def draw_bounding_box(img, boxes, scores):
    image_h, image_w, _ = img.shape
    for box, score in zip(boxes, scores):
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, (255, 255, 0), bbox_thick)
        cv2.putText(
            img, f"{score:.2f}", (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return img


def main_and_inter_iou(main_box, inter_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(main_box[0], inter_box[0])
    yA = max(main_box[1], inter_box[1])
    xB = min(main_box[2], inter_box[2])
    yB = min(main_box[3], inter_box[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if inter_area == 0:
        return 0, 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    main_box_area = abs(
        (main_box[2] - main_box[0]) * (main_box[3] - main_box[1]))
    inter_box_area = abs(
        (inter_box[2] - inter_box[0]) * (inter_box[3] - inter_box[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    main_iou = inter_area / float(main_box_area)
    all_iou = inter_area / float(inter_box_area + main_box_area - inter_area)

    # return the intersection over union value
    return main_iou, all_iou


def aspect_ratio_filter(box, aspect_ratio_list):
    p_one = box[0]
    p_two = box[1]
    p_three = box[3]

    euc_dis_x = ((p_one[0]-p_two[0])**2+(p_one[1]-p_two[1])**2)**0.5
    euc_dis_y = ((p_one[0]-p_three[0])**2+(p_one[1]-p_three[1])**2)**0.5

    logger.info(f"[TMP]: aspect ratio is {euc_dis_x/euc_dis_y}")

    if aspect_ratio_list[0] <= euc_dis_x/euc_dis_y <= aspect_ratio_list[1]:
        return True

    return False
