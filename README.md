# MobileFaceNet convert tool

This forked repo was from [LightFaceNet](https://github.com/SelinaFelton/LightFaceNet) . The script 'ckpt2tflite.py' is used to convert MobileFaceNet model from ckpt to tflite.

## Requirements

- tensorflow >= 1.13.1
- python 3.x

## Convert

$ python3 ckpt2tflite.py

The MobileFaceNet tflite model will be generated 'output/ckpt_best/mobilefacenet_best_ckpt_evl/MobileFaceNet_iter_14000.tflite'

## References

1. [LightFaceNet](https://github.com/SelinaFelton/LightFaceNet)