# Transformer-Based-Model-Learning

## Attention
Before learning transformer related model, it's useful to have a look at the attention mechanism first. 

Seq2seq is an Encoder–Decoder structured network with a sequence input and a sequence output. Both the input and output signal can be variable-length. 

![seq2seq](https://cdn-images-1.medium.com/max/1600/1*Ismhi-muID5ooWf3ZIQFFg.png)

We use "attention" layer to bridge information from decoder with encoder so that we can get a context vector which can be added to the original decoder hidden states vector.  

The following refers to: https://zhuanlan.zhihu.com/p/40920384

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

As for the score function, usually, there are there methods:

![equ8](https://pic3.zhimg.com/80/v2-129287642af2e34d7e9e0afea9ae766e_hd.jpg)

## Transformer

Refer to: https://blog.csdn.net/pipisorry/article/details/84946653

For the seq-to-seq problem, transformer uses scaled dot-product attention and multi-head attention instead of the traditional cnn or rnn architecture. 

Traditional rnn models like LSTM or GRU have certain drawbacks:
1. Sequencial calculation impedes parallel computing ability.
2. Lack of long dependency information.

Encoder = 6 * (self-attention + FFNN)
Decoder = 6 * (self-attention + encoder-decoder attention + FFNN), which uses outputs from encoder as inputs.

![t1](https://img-blog.csdnimg.cn/20181211141356770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BpcGlzb3JyeQ==,size_16,color_FFFFFF,t_70)

![t2](https://img-blog.csdnimg.cn/20181211142214247.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BpcGlzb3JyeQ==,size_16,color_FFFFFF,t_70)

Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.

![t3](https://img-blog.csdnimg.cn/20181210220041386.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BpcGlzb3JyeQ==,size_16,color_FFFFFF,t_70)

Now we dig more into each block in encoder and decoder:

### 1. Each encoder block
Multi-head self-attention mechanism + fully connected feed-forward network

#### Multi-head self-attention mechanism
Multi-head attention + residual connection + normalization

Multi-head attention:

Linear + scaled dot-product attention + concat + linear

Input: Query, Key, Value

Apply h different linear tranformations on Q,K,V, then concatenate all attention outputs. 

![equ1](https://www.zhihu.com/equation?tex=MultiHead%28Q%2C+K%2C+V%29+%3D+Concat%28head_1%2C+...%2C+head_h%29W%5EO+%5C%5C)

![equ2](http://www.zhihu.com/equation?tex=head_i+%3D+Attention%28QW_i%5EQ%2C+KW_i%5EK%2C+VW_i%5EV%29+%5C%5C)

![mh attention](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/MultiHead.png)

Note: In general attention mechanism, Q is the decoder hidden layer, K and V are the encoder hidden layer. While in self-attention, Q=K=V=sum of input embedding and positional embedding of encoder or decoder.

Scaled dot-product attention:

![equ3](http://www.zhihu.com/equation?tex=Attention%28Q%2C+K%2C+V%29+%3D+softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5C%5C)

Following the equation above, here is a clear example:

![self_attention](https://img-blog.csdnimg.cn/20181211144611901.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BpcGlzb3JyeQ==,size_16,color_FFFFFF,t_70)

Although it's called as multi-head, in practice, it's implemented by transposing and reshaping tensors, not separating tensors.

#### Position-wise feed-forward networks
If the output of multi-head attention is Z, then:

![equ4](https://www.zhihu.com/equation?tex=%5Ctext%7BFFN%7D%28Z%29+%3D+max%280%2C+ZW_1+%2Bb_1%29W_2+%2B+b_2+%5Ctag2)

### 2. Each decoder block
Same as encoder block, except that it has an attention sublayer. 

Notice that the input of whole decoder is the last output of decoder (output from position i-1 as the input of position i) and the output from encoder. Besides, we can apply parallel computing on encoder, however, as for decoder, just like rnn model, we can only use the last (and before) position's output as input. 

### 3. Positional Encoding 
Without position information, transformer is just a high-level word-bag model. In order to represent sequence information, positional encoding is added on the input embedding. 

## Bert

