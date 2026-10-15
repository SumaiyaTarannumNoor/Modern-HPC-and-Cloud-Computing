# Scenario 1: KV Cache + FlashAttention for SWM

## Flowchart

Video Frames
     |
     v
SWM Encoder / Feature Extractor
     |
     v
Visual Features
     |
     v
SWM Attention Module
     |
     v
Compute Q, K, V
     |
     v
Reuse valid cached K, V
or compute new K, V
     |
     v
FlashAttention
     |
     v
Updated Visual Features
     |
     v
SWM Output / Prediction

## Mathematical Formulation

Let \(X\) be the input features to an SWM attention layer.

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

Attention output:

\[
O=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

With KV caching:

\[
(K,V)_{\text{cached}}
\rightarrow
\operatorname{FlashAttention}(Q,K,V)
\]

**Description:** SWM reuses previously computed keys and values when the corresponding inputs and projection weights remain unchanged. FlashAttention computes the attention output efficiently using the available cached and newly computed values.

---

# Scenario 2: Attention Statistics Cache for SWM

## Flowchart

Video Frames
     |
     v
SWM Encoder / Feature Extractor
     |
     v
Visual Features
     |
     v
SWM Attention Module
     |
     v
Compute Q, K, V
     |
     v
Split K, V into blocks
     |
     v
Reuse valid cached statistics
or compute new statistics
     |
     v
Merge Block Statistics
     |
     v
Updated Visual Features
     |
     v
SWM Output / Prediction

## Mathematical Formulation

For query \(q\) and key-value block \(i\):

\[
S_i=\frac{qK_i^\top}{\sqrt{d_k}}
\]

Compute the block statistics:

\[
m_i=\max_j S_{ij}
\]

\[
\ell_i=\sum_j e^{S_{ij}-m_i}
\]

\[
u_i=\sum_j e^{S_{ij}-m_i}V_{ij}
\]

Cache:

\[
C_i=(m_i,\ell_i,u_i)
\]

Merge two block summaries:

\[
m'=\max(m_a,m_b)
\]

\[
\ell'=e^{m_a-m'}\ell_a+e^{m_b-m'}\ell_b
\]

\[
u'=e^{m_a-m'}u_a+e^{m_b-m'}u_b
\]

Final attention output:

\[
O=\frac{u'}{\ell'}
\]

**Description:** This approach caches intermediate attention statistics within an SWM attention layer. When the exact same query and key-value blocks recur with unchanged inputs, weights, masks, and positional information, the cached statistics can be reused and merged to calculate the output.

**Important:** “SWM” can refer to different models or architectures. These diagrams describe a generic SWM that contains a standard attention layer; the exact insertion point depends on which SWM implementation you are using.
