import cv2
import os
import time
import pandas as pd

VIDEO_LOCATION = 'data/'
FRAME_LOCATION = 'Data-image/'

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
      success, image = vidcap.read()
"""

def get_frames(vidcap, sec, fileLocation):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(fileLocation + str(sec) + " sec.jpg", image)  # save frame as JPG file
    return hasFrames

"""
def extract_frames():
    datafiles = os.listdir(VIDEO_LOCATION)
    f = 1
    start_time = time.process_time()
    for datafile in datafiles:
        print("f = ", f)
        f = f+1
        print(datafile)

        pathIn = VIDEO_LOCATION + datafile
        pathOut = FRAME_LOCATION + datafile[:-4]
        #print(pathIn)
        #print(pathOut)
        path = os.path.join(FRAME_LOCATION, datafile[:-4])
        os.mkdir(path)
        extractImages(pathIn, pathOut)
        """
        cap = cv2.VideoCapture(DEFAULT_DIRECTORY + '/' + datafile)
        fileLoc = FILE_LOCATION + datafile + '/'
        path = os.path.join(FILE_LOCATION, datafile)
        os.mkdir(path)
        sec = 0
        frameRate = 0.25
        success = get_frames(cap, sec, fileLoc)
        while success:
            sec = sec + frameRate
            sec = round(sec, 2)
            success = get_frames(cap, sec, fileLoc)
        print("Finished")
        """
    time_taken = time.process_time() - start_time
    print("Time taken:", time_taken)

def get_dataframe():
    images = []
    labels = []
    count = 0
    for f in os.listdir(FRAME_LOCATION):
        images.append(f)
        if "walk" in f:
            count = count+1
            labels.append("Walking")
        else:
            labels.append("Not Walking")
    df = pd.DataFrame()
    df['videos'] = images
    df['actions'] = labels
    df.to_csv('data.csv', header=True, index=False)


def main():
    extract_frames()
    print("Out")
    get_dataframe()

if __name__ == "__main__":
    main()