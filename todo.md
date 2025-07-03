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

5. Create Dual-Attention module class and instantiate in main. [ok]

6. Make code work for different values of T (also different between both sub-networks). [ok]

7. Create training.
    1. Check how to get QoE predictions' annotations. [ok]
        - Get the info and make it ready to use. [ok]
    2. Adapt trainer Class. [ok]

8. Update cuda memory management. [ok]
    - Check again the memory allocated to the backbone [ok]

9. Understand and implement correct batch processing. [ok]
    - For ResNet50: RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [4, 28, 3, 1080, 1920]
    - Use 3D ResNet. It does not run.

10. Find why there's different size labels and predictions for some particular cases. [ok]

11. Fix error in loss.backward() [ok]

13. For pre-processing, do as described in the paper:

    1. Since the position-encoding operation we use in the
    Transformer-encoder can only deal with fixed-length time
    series and the lengths of the videos in the datasets are not the
    same, we zero-pad these series to the longest video length in
    the dataset and mark the padded part, leaving out the marked
    part before the output. [ok]
    - confirm if it is not needed when processing one video at a time
        : src_key_padding_mask is only needed when dealing with batch_size > 0.

    2. For the input QoS features and the continuous and overall
    subjective experimental results used as labels for training and
    testing, we scale them to [0, 1] [14] by linear scaling:
    xscale = (x − xmin) / (xmax − xmin),
    where x means the value before scaling, xmax and xmin denote
    the maximum and minimum value of this feature or label in
    the whole dataset. [ok]

    3. Take subsamples from the labels (probably with the same pytorch Transforms) [ok]

14. Make sure loss function is correct and update it. [ok]

15. Resolve warnings. [N]

16. Use TensorBoard to visualize training progress and other activities [N]
    - Interpret correctly Tensorboard logs and adapt accordingly.

17. Implement Cross_Dataset update based on the paper 10.1109/TIP.2022.3190711 [N]

18. Test model only for continuous QoE for real-time inference testing. Use smaller chunk. [ok]
    - Adapt training learning rate.
    - Adapt data batching.

19. Speed up Dataset loading videos. [ok]
    - Cache dataset using torch.save and torch.load.

20. Rust Implementation. [NEXT_1]
    - Check if torch.jit.trace use is possible. [n1]
        - Check all data formats (specially the lists inputs and outputs handling) inside the model so it complies with jit.trace. Check if model is imported successfully.
        * Use https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/ as reference.
    - Make sute tch-rs works with Python .venv [ok]
    - Use Cuda in Rust. [ok]
    - Test inference function for custom model. [ok]
    - Implement evaluation metrics as in (22). [n0]
    - create real-time inference pipeline [n2]
        - create transformed input from videos. [ok]
        - create threading with tokio crate [n2]

22. Implement evaluation metrics to compare model performance with other papers [ok]
    . PLCC
    . SRCC
    . RMSE

23. Continue training from loaded model [NEXT_2]

24. Get data for Results chapter [NEXT_0]
    - Make plots for QoE predictions x QoE labels for a few videos. [ok]
    - Get data from validation set of metrics in (22) in Python. [ok]
        - for validation and training datasets. [n0]
    - Get best tensorboard data. [ok]
    - Ideally: get data from validation set of metrics in (22) in Rust to compare implementations. [n1]

25. Models only work in Rust with a fixed chunk size. [ok]
    - Implement variable size attention to mitigate this? When calculating in real-time, is this a problem?
        * Chunk size is defined on the model conversion by torch.jit.trace with chunk size input.

27. Understand training theory. [ ]