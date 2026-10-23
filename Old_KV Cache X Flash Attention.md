# Scenario 1: Standard KV Cache + FlashAttention

## Flowchart

Input sequence
      |
      v
Compute Q, K, V
      |
      v
Are K and V already cached?
      |
   +--+--+
   |     |
  Yes    No
   |     |
Reuse   Compute and cache K, V
   |     |
   +--+--+
      |
      v
FlashAttention(Q, K, V)
      |
      v
Attention Output

## Mathematical Formulation

\[
Q=X_QW_Q,\quad K=X_KW_K,\quad V=X_VW_V
\]

Standard attention:

\[
O=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

With KV caching, previously computed \(K\) and \(V\) are reused when their inputs and projection weights remain unchanged.

**Description:** FlashAttention computes the attention output efficiently, while the KV cache avoids recomputing reusable keys and values. This is exact up to floating-point numerical differences.

---

# Scenario 2: Custom Attention Statistics Cache

## Flowchart

Input Q, K, V
      |
      v
Split K, V into blocks
      |
      v
Compute block attention statistics
      |
      v
Are valid statistics cached for this exact Q and block?
      |
   +--+--+
   |     |
  Yes    No
   |     |
Reuse   Compute and cache statistics
   |     |
   +--+--+
      |
      v
Merge block statistics
      |
      v
Normalize and produce output

## Mathematical Formulation

For query \(q\) and key-value block \(i\):

\[
S_i=\frac{qK_i^\top}{\sqrt{d_k}}
\]

\[
m_i=\max(S_i)
\]

\[
\ell_i=\sum_j e^{S_{ij}-m_i}
\]

\[
u_i=\sum_j e^{S_{ij}-m_i}v_{ij}
\]

Cache the statistics:

\[
C_i=(m_i,\ell_i,u_i)
\]

Merge two cached or newly computed summaries:

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

**Description:** Instead of caching only \(K,V\), this scenario caches intermediate attention statistics and reuses them when the exact same query, key-value block, and computation conditions recur. The merged result is mathematically equivalent to standard attention, subject to floating-point rounding.

**Important condition:** A new query or changed keys, values, weights, masks, or positional information can invalidate the cached statistics.
