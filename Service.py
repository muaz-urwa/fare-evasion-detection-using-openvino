import argparse
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import pickle

from openvino.inference_engine import IENetwork, IECore, IEPlugin

from src.detector import Detector
from src.pose_estimator import HumanPoseEstimator
from src.track import Detection, Track
from src.utils import preprocess,  iou
from src.utils import draw_tracks


def run_demo(args):

    skip_frames = args.skip_frames
    out_fps = args.out_fps
    sigma_iou= args.sigma_iou
    log = args.log
    in_video_path = args.in_video_path
    device = args.device
    max_miss_frames = 3
    min_frame_th = 3
    
    video_name = in_video_path.split('/')[-1].split('.')[0]

    # setup experiment directory
    if not os.path.exists('runs'):
        os.makedirs('runs')
    exp_id = len(os.listdir('runs'))
    exp_dir = os.path.join('runs', 'exp_'+str(exp_id))
    os.mkdir(exp_dir)
    violation_dir = os.path.join(exp_dir, 'violations')
    os.mkdir(violation_dir)


    print("Experiment Directory: ", exp_dir)
    print('==== Configuration ====')
    print(args)


    # load models
    model_od = 'models/mobilenet_ssd/FP16/mobilenet-ssd.xml'
    mode_pose = 'models/pose_estimation/FP16/single-human-pose-estimation-0001.xml'
    cls_file = 'models/pose_classifier/classifier.sav'

    ie = IECore()
    detector_person = Detector(ie, path_to_model_xml=model_od,
                            device=device,
                            label_class=15)

    single_human_pose_estimator = HumanPoseEstimator(ie, path_to_model_xml=mode_pose,
                                                    device=device)

    classifier = pickle.load(open(cls_file, 'rb'))


    
    #read video file
    cap = cv2.VideoCapture(in_video_path)
    ret, frame = cap.read()

    # output video
    out = cv2.VideoWriter(os.path.join(exp_dir, video_name+'.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 
                          out_fps, (frame.shape[1],frame.shape[0] ))

    #time benchmarks
    total_time = 0
    detector_time = 0
    pose_time = 0
    classification_time = 0
    tracking_time = 0
    operation_count = 0


    tracks_active = []


    t_id = 1
    frame_i = 0
    while(cap.isOpened()):
        # read a frame from video
        ret, frame = cap.read() 

        frame_i +=1

        # if valid frame read
        if ret == True:

            # skip frames
            if frame_i % skip_frames == 0:

                operation_count += 1
                start_time = time.time()

                if log:
                    print("====== Frame id : ", str(frame_i))

                # detect person
                s = time.time()
                boxes = detector_person.detect(frame)
                detector_time += time.time() - s

                # extract pose
                s = time.time()
                key_points = [single_human_pose_estimator.estimate(frame, bbox) for bbox in boxes]
                pose_time += time.time() - s

                if log:
                    print("Detections : ", str(len(key_points)))


                # predict state and get detections
                s = time.time()
                detections_frame = []
                for box,k_p in zip(boxes, key_points):
                    features = preprocess(k_p)
                    state = classifier.predict(features)
                    det = Detection(box=box, state=state, frame=frame_i)
                    detections_frame.append(det)
                classification_time += time.time() - s

                dets = detections_frame
                
                # person tracking
                s = time.time()

                updated_tracks = []
                for track in tracks_active:

                    if len(dets) > 0:

                        best_match = max(dets, key=lambda x: iou(track.position, x.box))
                        if iou(track.position, best_match.box) >= sigma_iou:
                            track.update(best_match.box, best_match.state,frame_i, frame)
                            updated_tracks.append(track)

                            # remove from best matching detection from detections
                            del dets[dets.index(best_match)]

                    # if track was not updated
                    if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                        # finish track when the conditions are met
                        track.miss_track(frame_i)
                        if track.miss_count < max_miss_frames:
                            updated_tracks.append(track)

                # create new tracks
                new_tracks = []

                for det in dets:
                    new_tracks.append(Track(det.box, det.state, det.frame, frame_i, t_id, violation_dir))
                    t_id += 1

                tracks_active = updated_tracks + new_tracks

                tracking_time += time.time() - s

                if log:
                    print("Active Tracks : ", str(len(tracks_active)))

                valid_tracks = [t for t in tracks_active if t.frame_count() > min_frame_th]
                frame = draw_tracks(valid_tracks, frame)
                
                # save results
                out.write(frame) 
                total_time += time.time() - start_time

        else:
            break


    cap.release()
    
    print("======= FPS Report =======")
    print("Total fps: " +str(float(operation_count)/total_time ))
    print("Detector fps: " +str(float(operation_count)/detector_time ))
    print("Pose estimation fps: " +str(float(operation_count)/pose_time ))
    print("Pose classification fps: " +str(float(operation_count)/classification_time ))
    print("Person Tracker fps: " +str(float(operation_count)/tracking_time ))





def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_frames", type=int, required=False, default=2,
                        help="number of continuous frames to skip")

    parser.add_argument("--sigma_iou", type=int, default=0.5, required=False,
                    help="IOU threshold for person tracking")

    parser.add_argument("--out_fps", type=int,  default=3, 
                    help="fps of the output video")
    
    parser.add_argument("--log", type=bool, required=False, default=True, 
                    help="Print per frame logs")
    
    parser.add_argument("--in_video_path", required=False, default='data/videos/cusp_jump.mp4')

        
    parser.add_argument("--device", required=False, default='CPU',
                    help="Specify the target to infer on CPU or MYRIAD")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_demo(args)

