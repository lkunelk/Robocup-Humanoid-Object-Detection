# Undegraduate Thesis Project
Convolutional Neural Networks for Ball and Robot detection in Robocup Humanoid Kid-Size league.

![Alt text](outputs/sample.png?raw=true "Activations with bounding boxes")

## Download Dataset
images and corresponding label file should be placed together in one folder
https://imagetagger.bit-bots.de/images/imageset/835/


['../bit-bots-ball-dataset-2018/train/bitbots-set00-05',
'../bit-bots-ball-dataset-2018/train/sequences-jasper-euro-ball-1',
'../bit-bots-ball-dataset-2018/train/sequences-euro-ball-robot-1',
'../bit-bots-ball-dataset-2018/train/bitbots-set00-07',
'../bit-bots-ball-dataset-2018/train/bitbots-set00-04',
'../bit-bots-ball-dataset-2018/train/bitbots-set00-10',
'../bit-bots-ball-dataset-2018/train/imageset_352',
'../bit-bots-ball-dataset-2018/train/imageset_168',
'../bit-bots-ball-dataset-2018/train/bitbots-set00-08',
'../bit-bots-ball-dataset-2018/train/imageset_61',
'../bit-bots-ball-dataset-2018/train/sequences-misc-ball-1']
 = [os.path.join(valid_path, folder) for folder in os.listdir(valid_path)]

## Run model
TODO

## Train model
TODO

This work builds on top of Team Hamburg Bit Bots [model](https://robocup.informatik.uni-hamburg.de/wp-content/uploads/2018/06/2018_Speck_Ball_Localization.pdf).
