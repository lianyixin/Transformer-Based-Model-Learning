# Transformer-Based-Model-Learning

## Attention
Before learning transformer related model, it's useful to have a look at the attention mechanism first. 

Seq2seq is an Encoder–Decoder structured network with a sequence input and a sequence output. Both the input and output signal can be variable-length. 

![seq2seq](https://cdn-images-1.medium.com/max/1600/1*Ismhi-muID5ooWf3ZIQFFg.png)

We use "attention" layer to bridge information from decoder with encoder so that we can get a context vector which can be added to the original decoder hidden states vector.  

The following comes from: https://zhuanlan.zhihu.com/p/40920384

![attention1](https://pic4.zhimg.com/80/v2-8ddf993a95ee6e525fe2cd5ccd49bba7_hd.jpg)

Input: x = (x1, x2, ..., xt_x), Output: y = (y1, y2, ..., yt_y)

For each encoder cell, the hidden state output at time step t:

![equ1](https://www.zhihu.com/equation?tex=h_t+%3D+RNN_%7Benc%7D%28x_t%2C+h_%7Bt-1%7D%29)

For each decoder cell, the hidden state output at time step t:

![equ2](https://www.zhihu.com/equation?tex=s_t+%3D+RNN_%7Bdec%7D%28%5Chat%7By_%7Bt-1%7D%7D%2Cs_%7Bt-1%7D%29)

For each hidden state in decoder (at time step i), we calculate scores with each hidden state in encoder (at time step j), which can be represented as weight:

![equ3](https://www.zhihu.com/equation?tex=e_%7Bij%7D+%3D+score%28s_i%2C+h_j%29)

Then we can get the weight of each hidden state in encoder for any hidden state j in decoder and get the weighted average context vector ci：

![equ4](https://www.zhihu.com/equation?tex=%5Calpha_%7Bij%7D+%3D+%5Cfrac%7Bexp%28e_%7Bij%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BT_x%7Dexp%28e_%7Bik%7D%29%7D)

![equ5](https://www.zhihu.com/equation?tex=c_i+%3D+%5Csum_%7Bj%3D1%7D%5E%7BT_x%7D+%5Calpha_%7Bij%7Dh_j)

Finally, we concatenate ci and si, and apply softmax on this new vector:

![equ6](https://www.zhihu.com/equation?tex=%5Chat%7Bs_t%7D+%3D+tanh%28W_c%5Bc_t%3Bs_t%5D%29)

![equ7](https://www.zhihu.com/equation?tex=p%28y_t%7Cy_%7B%3Ct%7D%2Cx%29+%3D+softmax%28W_s%5Chat%7Bs_t%7D%29)
