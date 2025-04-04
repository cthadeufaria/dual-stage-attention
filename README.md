# dual-stage-attention-QoE

Convolutional Neural Network Based Quality of Experience Estimation Algorithm written in Python.

Implementation of the Dual-Stage Attention algorithm described in the paper: "Continuous and Overall Quality of Experience Evaluation for Streaming Video Based on Rich Features Exploration and Dual-Stage Attention".

Paper available @ DOI: 10.1109/TCSVT.2024.3418941


TODO:

1. Understand what's the video chunk of T frames.
    a. get T from Dataset 'inputs' if possible. [ok]
    b. update 'main' to handle T. [ok]
    c. fix global average pooling from backbone. [ok]
    d. fix fully connected networks to use T correctly. [ok]
    e. fix short-time regression module to use T correctly. [ok]

2. Create Long-Time Temporal Regression Module. [ok]

3. Create Cross-Feature Attention Module. [ok]

4. Update Dataset class to get QoS features. [ok]

5. Create Dual-Attention module class and instantiate in main. [O]

6. Make code work for different values of T (also different between both sub-networks). [ ]

7. Update code to get continuous video chunks from all videos from the Dataset. [ ]

- For training, see https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/