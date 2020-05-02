# CSCE 689: Action Detection

This project is part of the coursework for CSCE 689 - Advanced Deep Learning. For a given target action - walking, deep learning techniques are used to detect the action in the given video.
 
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/en/latest/) to install the necessary packages.

## Data
The data is stored here: [HMDB](https://drive.google.com/drive/folders/1H8FbrRoRBKbd1fMGlNnnzDOLNzouuxNX?usp=sharing), [NIXMAS](https://drive.google.com/drive/folders/1w8Kkoz2xdy0Fy_Mpa0B44wCwEsz8xKIo?usp=sharing). This contains the training videos, frames of the videos, test samples, generated landmarks using openpose, json files.

1. If using google [colab](colab.research.google.com),mount the drive and use the files. Use colab with gpu runtime.
2. If running locally or in cluster, download the data and update the data locations appropriately in the code. Use gpu to train the models.

## Files

### utility/
1. data_loader.py - contains functions to load the train and test data and for preprocessing the images.
2. model.py - models used in the project. Code for 3DCNN, CRNN and Resnet + RNN.
3. videoToFrame.py - to convert the training data videos to frames. Train videos are in drive folder. 

```python
python ./videoToFrame.py 
```

4. splitVideo.sh - to split the test videos to multiple components to run prediction. 

```bash
./splitVideo --<video_file>
```

5. openpose.ipynb - to generate landmarks using openpose. Run the cells in the notebook. Enter the correct location of the video files.

### model/
Program to train the models.

```python
python ./<model_name>.py 
```

The trained models are included in the drive link. 

## Instruction to run prediction code
1. Load the test files in the google drive.
2. Split the test videos to multiple samples using splitVideo.sh.
3. Convert video to frames.
4. Open the prediction_code.ipynb in utility/. Enter location of the test video images and run the code. The prediction results will be stored in .json format.
