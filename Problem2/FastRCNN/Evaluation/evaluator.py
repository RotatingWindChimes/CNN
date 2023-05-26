from Evaluation.voc_eval import voc_eval
import os


class Evaluator(object):
    def __init__(self):
        self.det_img_name = list()
        self.det_additional = list()
        self.det_boxes = list()
        self.det_labels = list()
        self.det_scores = list()

    def get_info(self, info):
        (pred_boxes, pred_labels, pred_scores, img_names, additional_info) = info
        self.det_img_name.append(img_names)
        self.det_additional.append(additional_info)
        self.det_boxes.append(pred_boxes)
        self.det_labels.append(pred_labels)
        self.det_scores.append(pred_scores)

    def evaluate(self, test_root):
        mAP = voc_eval(test_root, self.det_img_name, self.det_additional, self.det_boxes, self.det_scores,
                           self.det_labels)
        return mAP