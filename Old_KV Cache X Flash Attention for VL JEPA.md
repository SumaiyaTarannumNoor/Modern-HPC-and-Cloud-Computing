# Scenario 1: KV Cache + FlashAttention for VL-JEPA

## Flowchart

Video Frames
     |
     v
Visual Encoder
     |
     v
Visual Representations
     |
     v
VL-JEPA Predictor
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
Predicted Representations
     |
     v
Compare with Target Encoder Representations
     |
     v
Training Loss

## Mathematical Formulation

Let \(X_v\) represent visual features entering the predictor.

\[
Q=X_vW_Q,\quad K=X_vW_K,\quad V=X_vW_V
\]

The predictor's attention output is:

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

**Description:** In VL-JEPA, previously computed keys and values can be reused when the corresponding predictor inputs and projection weights are unchanged. FlashAttention then computes the attention output efficiently. The resulting predicted representations are compared with the target representations to calculate the training loss.

---

# Scenario 2: Attention Statistics Cache for VL-JEPA

## Flowchart

Video Frames
     |
     v
Visual Encoder
     |
     v
Visual Representations
     |
     v
VL-JEPA Predictor
     |
     v
Compute Q, K, V
     |
     v
Split K, V into blocks
     |
     v
Reuse valid cached attention statistics
or compute new statistics
     |
     v
Merge Block Statistics
     |
     v
Predicted Representations
     |
     v
Compare with Target Encoder Representations
     |
     v
Training Loss

## Mathematical Formulation

For predictor query \(q\) and key-value block \(i\):

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

Merge block summaries using:

\[
m'=\max(m_a,m_b)
\]

\[
\ell'=e^{m_a-m'}\ell_a+e^{m_b-m'}\ell_b
\]

\[
u'=e^{m_a-m'}u_a+e^{m_b-m'}u_b
\]

The predictor attention output is:

\[
O=\frac{u'}{\ell'}
\]

**Description:** Instead of caching only keys and values, this approach caches intermediate attention statistics inside the VL-JEPA predictor. If the exact same query and key-value blocks recur under unchanged computation conditions, their statistics can be reused and merged to obtain the attention output.

**Important:** During training, predictor weights and representations may change between steps, invalidating cached statistics. Reuse is valid only when the relevant inputs, weights, masks, and positional information are unchanged.
