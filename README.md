# dual-stage-attention-QoE

Convolutional Neural Network Based Quality of Experience Estimation Algorithm written in Python.

Implementation of the Dual-Stage Attention algorithm described in the paper: "Continuous and Overall Quality of Experience Evaluation for Streaming Video Based on Rich Features Exploration and Dual-Stage Attention".

Paper available @ DOI: 10.1109/TCSVT.2024.3418941


TODO:

1. Understand what's the video chunk of T frames.
    a. get T from Dataset 'inputs' if possible. [ok]
    b. update 'main' to handle T. [ok]
    c. fix global average pooling from backbone. [ok]
    d. fix fully connected networks to use T correctly. [ ]
    e. fix short-time regression module to use T correctly. [ ]

2. Update Dataset class to get QoS features.

3. Structure TODOs in the code here to plan tasks.

4. Update code to get all video chunks from all videos from the Dataset. Now it only takes the 1st chunk from the 1st video.

- For training, see https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/