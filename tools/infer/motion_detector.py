import cv2
import numpy as np


class MotionDetector:
    def __init__(self, bg_history=10, area_list=[2000, 2000000],
                 wh_ratio_list=[0.5, 10], nms_thresh=0.1):
        self._area_list = area_list
        self._wh_ratio_list = wh_ratio_list
        self._nms_thresh = nms_thresh

        self._fgbg = cv2.createBackgroundSubtractorMOG2(history=bg_history)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _calculate(self, bound, mask):
        x, y, w, h = bound
        area = mask[y:y+h, x:x+w]
        pos = area > 0 + 0
        score = np.sum(pos)/(w*h)

        return score

    def _py_cpu_nms(self, dets, thresh):
        y1 = dets[:, 1]
        x1 = dets[:, 0]
        y2 = y1 + dets[:, 3]
        x2 = x1 + dets[:, 2]

        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1.0, xx2 - xx1 + 1)
            h = np.maximum(1.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]

            order = order[inds + 1]

        return keep

    def _area_filter(self, cnts, mask):
        bounds = []

        for c in cnts:
            contour_area = cv2.contourArea(c)
            if self._area_list[0] < contour_area < self._area_list[1]:
                bound = cv2.boundingRect(c)
                wh_ratio = bound[2] / bound[3]
                if self._wh_ratio_list[0] <= wh_ratio <= self._wh_ratio_list[1]: # noqa
                    bounds.append(bound)

        if len(bounds) == 0:
            return []

        scores = [self._calculate(b, mask) for b in bounds]
        bounds = np.array(bounds)
        scores = np.expand_dims(np.array(scores), axis=-1)
        keep = self._py_cpu_nms(np.hstack([bounds, scores]), self._nms_thresh)

        return bounds[keep]

    def detect(self, frame):
        fgmask = self._fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounds = self._area_filter(contours, fgmask)

        bound_result = []
        for bound in bounds:
            xmin = bound[0]
            ymin = bound[1]
            xmax = bound[0] + bound[2]
            ymax = bound[1] + bound[3]
            bound_result.append([xmin, ymin, xmax, ymax])

        return bound_result
