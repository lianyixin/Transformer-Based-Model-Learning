# Transformer-Based-Model-Learning

## Attention
Before learning transformer related model, it's useful to have a look at the attention mechanism first. 

Seq2seq is an Encoder–Decoder structured network with a sequence input and a sequence output. Both the input and output signal can be variable-length. 

![seq2seq](https://cdn-images-1.medium.com/max/1600/1*Ismhi-muID5ooWf3ZIQFFg.png)

We use "attention" layer to bridge information from decoder with encoder so that we can get a context vector which can be added to the original decoder hidden states vector.  

The following comes from: https://zhuanlan.zhihu.com/p/40920384

![attention1](https://pic4.zhimg.com/80/v2-8ddf993a95ee6e525fe2cd5ccd49bba7_hd.jpg)

Input: x = (x1, x2, ..., xt), Output: y = (y1, y2, ..., yt)

For each encoder cell:
![equ1](https://www.zhihu.com/equation?tex=h_t+%3D+RNN_%7Benc%7D%28x_t%2C+h_%7Bt-1%7D%29)

For each decoder cell:
![equ2](https://www.zhihu.com/equation?tex=s_t+%3D+RNN_%7Bdec%7D%28%5Chat%7By_%7Bt-1%7D%7D%2Cs_%7Bt-1%7D%29)
