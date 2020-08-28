import numpy as np
import cv2


def iou(bbox1, bbox2):

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union



def preprocess_bbox(bbox, image):
    aspect_ratio = 0.75
    bbox[0] = np.clip(bbox[0], 0, image.shape[0] - 1)
    bbox[1] = np.clip(bbox[1], 0, image.shape[0] - 1)
    x2 = np.min((image.shape[1] - 1, bbox[0] + np.max((0, bbox[2] - 1))))
    y2 = np.min((image.shape[0] - 1, bbox[1] + np.max((0, bbox[3] - 1))))

    bbox = [bbox[0], bbox[1], x2 - bbox[0], y2 - bbox[1]]

    cx_bbox = bbox[0] + bbox[2] * 0.5
    cy_bbox = bbox[1] + bbox[3] * 0.5
    center = np.array([np.float32(cx_bbox), np.float32(cy_bbox)])

    if bbox[2] > aspect_ratio * bbox[3]:
        bbox[3] = bbox[2] * 1.0 / aspect_ratio
    elif bbox[2] < aspect_ratio * bbox[3]:
        bbox[2] = bbox[3] * aspect_ratio

    s = np.array([bbox[2], bbox[3]], np.float32)
    scale = s * 1.25

    return center, scale


def extract_keypoints(heatmap, min_confidence=-100):
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    if heatmap[ind] < min_confidence:
        ind = (-1, -1)
    else:
        ind = (int(ind[1]), int(ind[0]))
    return heatmap[ind[1]][ind[0]], ind


def affine_transform(pt, t):
        transformed_point = np.dot(t, [pt[0], pt[1], 1.])[:2]
        return transformed_point


class TransformedCrop(object):
    def __init__(self, input_height=384, input_width=288, output_height=48, output_width=36):
        self._num_keypoints = 17
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

    def __call__(self, img, bbox):
        c, s = preprocess_bbox(bbox, img)
        trans, _ = self.get_trasformation_matrix(c, s, [self.input_width, self.input_height])
        transformed_image = cv2.warpAffine(img, trans, (self.input_width, self.input_height), flags=cv2.INTER_LINEAR)
        rev_trans = self.get_trasformation_matrix(c, s, [self.output_width, self.output_height])[1]

        return rev_trans, transformed_image.transpose(2, 0, 1)[None, ]

    @staticmethod
    def get_trasformation_matrix(center, scale, output_size):

        w, h = scale
        points = np.zeros((3, 2), dtype=np.float32)
        transformed_points = np.zeros((3, 2), dtype=np.float32)

        transformed_points[0, :] = [output_size[0] * 0.5, output_size[1] * 0.5]
        transformed_points[1, :] = [output_size[0] * 0.5, output_size[1] * 0.5 - output_size[0] * 0.5]
        transformed_points[2, :] = [0, output_size[1] * 0.5]

        shift_y = [0, - w * 0.5]
        shift_x = [- w * 0.5, 0]

        points[0, :] = center
        points[1, :] = center + shift_y
        points[2, :] = center + shift_x

        rev_trans = cv2.getAffineTransform(np.float32(transformed_points), np.float32(points))

        trans = cv2.getAffineTransform(np.float32(points), np.float32(transformed_points))

        return trans, rev_trans


def plot_keypoints(key_points, img):
    colors = [(0, 0, 255),
              (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
              (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
              (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
              (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]
    for id_kpt, kpt in enumerate(key_points):
            cv2.circle(img, (int(kpt[0]), int(kpt[1])), 10, colors[id_kpt], -1)

    return img

def preprocess(k_p):
    k_p = np.array(k_p)

    min_vals = np.min(k_p, axis=0)
    max_vals = np.max(k_p, axis=0)
    
    k_p = (k_p - min_vals) / (max_vals - min_vals)

    k_p = k_p.flatten(order='F')
    k_p = k_p.reshape(1,34)

    return k_p


def draw_tracks(tracks, frame):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # fontScale 
    fontScale = 1
    thickness = 2
    box_thickness = 2    
    
    for track in tracks:

        if track.state == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        p1 = (track.position[0],track.position[1])
        p2 = (track.position[2],track.position[3])

        frame = cv2.rectangle(frame, p1, p2 
                                , color, box_thickness)

        frame = cv2.putText(frame, str(track.t_id), p1, font,  
                fontScale, color, thickness, cv2.LINE_AA) 

    return frame