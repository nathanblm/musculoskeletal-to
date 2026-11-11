RTMW3D inference components and model configuration are vendored from
https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose3d
under the accompanying Apache 2.0 LICENSE.

Local modification: pose_estimator.py computes the default principal point
as (width / 2, height / 2), correcting upstream's swapped image dimensions.
The upstream default focal length and mean root-depth assumptions are retained.
These monocular predictions are not calibrated metric measurements.
