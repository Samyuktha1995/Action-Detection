import cv2
import os
import time
import pandas as pd

from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

VIDEO_LOCATION = '/content/gdrive/My Drive/HDMI_Data/HDMI_data/'
FRAME_LOCATION = '/content/gdrive/My Drive/HDMI_Data/Data_image/'

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(pathOut + "/frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
      success, image = vidcap.read()
      if count == 15:
          break

def extract_frames():
    datafiles = os.listdir(VIDEO_LOCATION)
    f = 1
    start_time = time.process_time()
    for datafile in datafiles:
        print("f = ", f)
        f = f+1
        print(datafile)

        pathIn = VIDEO_LOCATION + datafile
        pathOut = FRAME_LOCATION + datafile[:-4] + '/'
        print(pathIn)
        print(pathOut)
        path = os.path.join(FRAME_LOCATION, datafile[:-4])
        os.mkdir(path)
        extractImages(pathIn, pathOut)

    time_taken = time.process_time() - start_time
    print("Time taken:", time_taken)

def main():
    extract_frames()
    print("Out")

if __name__ == "__main__":
    main()