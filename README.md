# Video Prediction Models in PyTorch

This project contains PyTorch implementations of the Convolutional Dynamic Neural Advection (CDNA), Spatial Transformer Predictor (STP), and Dynamic Neural Advection (DNA) models, designed for the task of video prediction. These models aim to predict future frames in a video sequence given past frames, focusing on learning physical interactions in an unsupervised manner.

## Reference Paper

The models implemented in this project are based on the research presented in the paper:

- Chelsea Finn, Ian Goodfellow, and Sergey Levine, "Unsupervised Learning for Physical Interaction through Video Prediction," in *Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS 2016)*. [https://arxiv.org/abs/1605.07157](https://arxiv.org/abs/1605.07157)

This work introduces innovative neural network architectures for video prediction, which are capable of learning to predict future video frames in an unsupervised fashion. The models learn physical interaction dynamics by anticipating the future frames in video sequences, making them useful for a variety of applications in robotics, autonomous navigation, and interactive systems.
