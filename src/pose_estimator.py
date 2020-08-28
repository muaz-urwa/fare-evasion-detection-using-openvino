import numpy as np
import cv2
import os
from src.utils import TransformedCrop, extract_keypoints
from src.utils import affine_transform

class HumanPoseEstimator(object):
    def __init__(self, ie, path_to_model_xml, scale=None, thr=-100, device='CPU'):
        
        self.model = ie.read_network(path_to_model_xml, os.path.splitext(path_to_model_xml)[0] + '.bin')

        assert len(self.model.input_info) == 1, "Expected 1 input blob"

        assert len(self.model.outputs) == 1, "Expected 1 output blob"

        self._input_layer_name = next(iter(self.model.input_info))
        self._output_layer_name = next(iter(self.model.outputs))
        self.CHANNELS_SIZE = 3
        self.OUTPUT_CHANNELS_SIZE = 17

        assert len(self.model.input_info[self._input_layer_name].input_data.shape) == 4 and \
               self.model.input_info[self._input_layer_name].input_data.shape[1] == self.CHANNELS_SIZE,\
               "Expected model input blob with shape [1, 3, H, W]"

        assert len(self.model.outputs[self._output_layer_name].shape) == 4 and \
               self.model.outputs[self._output_layer_name].shape[1] == self.OUTPUT_CHANNELS_SIZE,\
            "Expected model output shape [1, %s, H, W]" % (self.OUTPUT_CHANNELS_SIZE)

        self._ie = ie
        self._exec_model = self._ie.load_network(self.model, device)
        self._scale = scale
        self._thr = thr

        _, _, self.input_h, self.input_w = self.model.input_info[self._input_layer_name].input_data.shape
        _, _, self.output_h, self.output_w = self.model.outputs[self._output_layer_name].shape
        self._transform = TransformedCrop(self.input_h, self.input_w, self.output_h, self.output_w)
        self.infer_time = -1

    def _preprocess(self, img, bbox):
        return self._transform(img, bbox)

    def _infer(self, prep_img):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_name: prep_img})
        self.infer_time = ((cv2.getTickCount() - t0) / cv2.getTickFrequency())
        return output[self._output_layer_name][0]

    @staticmethod
    def _postprocess(heatmaps, rev_trans):
        all_keypoints = [extract_keypoints(heatmap) for heatmap in heatmaps]
        all_keypoints_transformed = [affine_transform([kp[1][0], kp[1][1]], rev_trans) for kp in all_keypoints]

        return all_keypoints_transformed

    def estimate(self, img, bbox):
        rev_trans, preprocessed_img = self._preprocess(img, bbox)
        heatmaps = self._infer(preprocessed_img)
        keypoints = self._postprocess(heatmaps, rev_trans)
        return keypoints