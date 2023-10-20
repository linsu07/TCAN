# TCAN
Temporal Convolutional Attention-based Network 
## overview
  Predicting the future based on past temporal sequences is highly appealing, and one application of this is stock trend/price prediction. However, existing models like "LSTM" tend to forget information with long sequences, while "Transformer" focuses on interesting points but lacks an overview of the historical trend. To address these limitations, TCAN (Temporal Convolutional Attention-based Network) is proposed as a better choice. TCAN incorporates dilated convolution, self-attention, and multiple layers. The multi-layer dilated convolution captures the overall situation, while self-attention provides a better understanding of each position. The concept of TCAN is described in the paper "Temporal Convolutional Attention-based Network For Sequence Modeling," which is listed in this repository.
## features
  + In addition to the features described in the paper, a new feature called "local attention" is added to the model. This is because the TCAN model here is specifically designed for stock trend prediction, where the situation of one day is highly related to the surrounding days.
