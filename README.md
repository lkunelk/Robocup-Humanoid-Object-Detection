# Undegraduate Thesis Project
Enabling Ball and Robot detection using Convolutional Neural Networks for soccer playing humanoids.

include pic here

![Alt text](outputs/sample.png?raw=true "Activations with bounding boxes")

## Dataset
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
 
This work builds on top of Team Hamburg Bit Bots [model](https://robocup.informatik.uni-hamburg.de/wp-content/uploads/2018/06/2018_Speck_Ball_Localization.pdf).
