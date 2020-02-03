import cv2
import time
import os
import cvlib
import numpy as np

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        count = video_length
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def get_single_frame(file_name,output_loc, face = False):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Start capturing the feed
    cap = cv2.VideoCapture(file_name)
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, (1000))
        ret, frame = cap.read()
        if face:
            faces, chance = cvlib.detect_face(frame)
            faces[0][1] = np.clip(faces[0][1],0,len(frame))
            faces[0][3] = np.clip(faces[0][3], 0, len(frame))
            faces[0][0] = np.clip(faces[0][0], 0, len(frame[0]))
            faces[0][2] = np.clip(faces[0][2], 0, len(frame[0]))
            print(frame)
            frame = frame[faces[0][1]:faces[0][3],faces[0][0]:faces[0][2]]
            print(frame)
            print(faces,chance)
        cv2.imwrite(output_loc +"\\"+ file_name.split('\\')[-1].rstrip(".avi") + "_1.jpg", frame)
        # cap.set(cv2.CAP_PROP_POS_MSEC, (500))
        # ret, frame = cap.read()
        # cv2.imwrite(output_loc +"\\"+ file_name.split('\\')[-1].rstrip(".avi") + "_2.jpg", frame)
        # cap.set(cv2.CAP_PROP_POS_MSEC, (1000))  # added this line
        # ret, frame = cap.read()
        # cv2.imwrite(output_loc +"\\"+ file_name.split('\\')[-1].rstrip(".avi") + "_3.jpg", frame)
        # #print(output_loc +"\\"+ file_name.split('\\')[-1].rstrip(".mov") + ".jpg")
        cap.release()
        break




def dir_to_frames(input_loc, output_loc, face = False):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    for file in os.listdir(input_loc):
            get_single_frame(os.path.join(input_loc,file),output_loc,face)



def dataset_to_frames(loc, output_loc,face=False):
    print(loc)
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    for path in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,path)):
            try:
                os.mkdir(os.path.join(output_loc,path))
            except OSError:
                pass
            dataset_to_frames(os.path.join(loc,path),os.path.join(output_loc,path),face)
        else:
            dir_to_frames(loc,output_loc,face)
            break
            print("dir done")


input_loc = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\casia-fasd\train'
output_loc = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\casia-fasd\train_frames_faces'
dataset_to_frames(input_loc, output_loc,True)