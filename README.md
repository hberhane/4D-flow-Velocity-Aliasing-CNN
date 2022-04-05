# 4D-flow-Velocity-Aliasing-CNN
These are the training and testing scripts for a CNN that detects and corrects velocity aliasing in 4D flow MRI

The current implmenatation of these scripts uses tensorflow==1.12.0

If people would like, I can rewrite to tensorflow 2.

Also the training script automatically generates the simulated velocity aliasing, the input are supposed to be just the velocity data, one time-frame at a time.

The testing script assumes that aliasing is already present in the data. So if testing on simuilated aliasing,
make simulated aliasing prior to inputting to CNN via testing script.

Also currently the scripts take tfrecords as inputs and outputs are mat files.

The CNN architecture has been described previously here:  https://doi.org/10.1002/mrm.28257

Will hopefuilly update soon with aorta segmentation scripts.

Please feel free to reach out via email with questions, comments, or suggestions :)
Email: hberhane@u.northwestern.edu
