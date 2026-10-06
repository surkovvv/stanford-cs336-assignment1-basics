# Academic checkpoint — October 6, 2026

This is a record of my answers to selected **external exam questions** after working through CS336 Assignment 1 / TinyStories §7.3. The question descriptions below are paraphrases; the linked exams contain the original wording. “My answer” records what I submitted before checking solutions. The review is separate.

## MIT 6.390, Spring 2025 final, Problem 4: Transforming Transformers

[Exam](https://introml.mit.edu/_static/fall25/midterm2/review/final-spring2025.pdf) · [Official solutions](https://introml.mit.edu/_static/fall25/midterm2/review/final-spring2025-solutions.pdf)

Setup: one attention head, **no positional encoding**. Input length is `n`, token dimension is `d`, and query/key/value dimension is `d_k`. Attention is row-wise `softmax(QKᵀ / √d_k)`. In part b the initial sequence is `[Lucy, ate, her, breakfast]`, with the following score matrix (rows and columns follow that order):

```text
QKᵀ = [[ 3, 1,  3,  0],
       [ 0, 3,  3,  1],
       [-1, 1,  2, -2],
       [ 0, 2, -2, -1]]
```

The candidates for b.i are:

```text
A1 = [[0.42, 0.05, 0.17, 0.36],
      [0.37, 0.09, 0.15, 0.39],
      [0.47, 0.02, 0.40, 0.11],
      [0.37, 0.16, 0.11, 0.36]]
A2 = [[0.4576, 0.0619, 0.4576, 0.0228],
      [0.0228, 0.4576, 0.4576, 0.0619],
      [0.0347, 0.2562, 0.6964, 0.0128],
      [0.1125, 0.8310, 0.0152, 0.0414]]
A3 = [[ 0.2940, -0.3831,  0.3229,  0.0000],
      [ 0.1766,  0.2692,  0.1824,  0.3717],
      [-0.3612,  0.1437,  0.2968, -0.1983],
      [ 0.0000,  0.8541, -0.0655, -0.0804]]
A4 = [[0.4, 0.3, 0.2, 0.1],
      [0.3, 0.4, 0.1, 0.2],
      [0.2, 0.1, 0.5, 0.2],
      [0.1, 0.2, 0.2, 0.5]]
A5 = [[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]]
```

For c, the mask is `M = [[0, -∞, -∞, -∞], [0, 0, -∞, -∞], [0, 0, 0, -∞], [0, 0, 0, 0]]`, added to the scores before softmax. For d, `X = I₄` and `K = [[-2, -2], [2, -1], [-3, 1], [1, 3]]`.

| Part | Question / condition | My answer | Review |
| --- | --- | --- | --- |
| a.i | What is the shape of `W_q`? | `d × d_k`, by definition. | Correct. |
| a.ii | Which of `W_q`, `W_v`, `K`, `Q`, and `A` grow when `n` grows? | `A` grows because its last two axes are `n × n`. | Incomplete: `Q` and `K` also gain rows; `W_q` and `W_v` keep their shapes. |
| a.iii | Must `q_i − v_i = q_j − v_j` hold for two distinct input tokens? | False: the difference depends on the input token through the query and value weights. | Correct conclusion. More precisely, equality would require `(W_q − W_v)ᵀ(x_i − x_j) = 0`, which is not guaranteed. Distinct inputs do not rule out equality in every special case. |
| b.i | Choose a possible attention matrix from five candidates. | `A2`; its entries have the softmax ordering and are nonnegative. | Matches the official key. Positive entries alone are insufficient; the row-wise exponential normalization must also match. |
| b.ii | Can `d_k` be deduced from the observed `QKᵀ`? | No: multiplying `Q ∈ ℝ^(n×d_k)` by `Kᵀ ∈ ℝ^(d_k×n)` leaves an `n × n` matrix. | Correct according to the official key. |
| b.iii | Compare attention `Lucy → ate` with `ate → Lucy`. | `Lucy → ate` is larger, from `A2`. | Correct. The two score rows have the same multiset, hence the same softmax denominator, while the relevant scores are `1` and `0`. |
| b.iv | Repeat the comparison after permuting the sequence to `[her, breakfast, ate, Lucy]`, keeping weights fixed. | `Lucy → ate` remains larger; only the positions change. | Correct because this model has no positional encoding. |
| b.v | Repeat after replacing `breakfast` with `dinner`. | Not enough information. I attributed this to the mask / unseen word. | Correct result, wrong reason: no mask is introduced yet. The new token produces unknown attention scores and may change the two softmax denominators differently. |
| b.vi | Repeat after extending the sequence with `too, late`. | Not enough information; I said we do not know the new word values/embeddings. | Correct result. The missing information is the new query/key scores and their effect on the softmax denominators; `V` does not determine attention weights. |
| c | Apply a causal mask to the original sequence; which token receives the most attention from `Lucy`? | `Lucy`, because the other tokens are masked for her. | Correct. |
| d | With four one-hot inputs, `X = I₄`, and the given `K`, recover `W_k`. | I used `K = XW_kᵀ`, obtaining `W_k = Kᵀ`. | Transposed relative to the exam convention: `K = XW_k`, so `W_k = K`, shape `4 × 2`. |

For b.i, the official solution calls `A2` the result for `d_k = 1`. There is a mathematical inconsistency in the exam data: the displayed `QKᵀ` has rank 4, so it cannot be factored through one-dimensional `Q` and `K`. The review above follows the intended official answer.

## Stanford CME 295, Fall 2025 midterm: selected questions

[Exam](https://cme295.stanford.edu/exams/midterm.pdf)

The multiple-choice conditions and options, paraphrased:

| Part | Prompt and choices |
| --- | --- |
| I.1 | Advantage of subword over whole-word tokenization: **A** no vocabulary; **B** fewer unknown words through reusable word parts; **C** interpretable embeddings without training; **D** no training data. |
| II.4 | Main effect of RoPE: **A** add sinusoidal vectors to token embeddings; **B** mask distant tokens; **C** uniformly weaken values with distance; **D** position-dependent rotation of query/key two-dimensional components. |
| II.5 | Common LLM positional method: **A** RoPE within attention; **B** learned absolute token-position vectors; **C** only fixed sinusoidal token-position vectors; **D** relative-position bias applied only to values. |
| II.7 | Location of a causal mask: **A** query-key scores before softmax; **B** values after softmax; **C** token embeddings before attention; **D** final output logits. |
| II.8 | Main role of a feedforward sublayer: **A** combine sequence order; **B** reduce attention heads; **C** replace positional encoding; **D** per-token nonlinearity and feature-channel mixing. |

I.9 is a free-response question asking for the self-attention equation and the roles of `Q`, `K`, and `V`.

| Part | Question / condition | My answer | Review |
| --- | --- | --- | --- |
| I.1 | Main benefit of subword tokenization over whole-word tokenization; choose one option. | **B:** it reduces out-of-vocabulary issues by reusing frequent word parts. | Correct. |
| I.9 | Write the self-attention formula and explain `Q`, `K`, and `V`. | `softmax(QKᵀ / √d_k)V`, with `Q = XW_q` and `X ∈ ℝ^(n×d)`. Queries look for matching keys; values are the content retrieved through those matches. | Correct conceptually. Keep matrix orientation consistent: with `Q = XW_q`, `W_q ∈ ℝ^(d×d_k)`. |
| II.4 | What does RoPE do? | **D:** rotate query and key components using position-dependent two-dimensional blocks. | Correct. |
| II.5 | Which listed positional method is commonly used in LLMs? | **A:** RoPE inside attention. | Correct for the exam. |
| II.7 | Where is the causal mask applied? | **A:** to `QKᵀ` scores before softmax. | Correct. |
| II.8 | What is the feedforward sublayer's main role? | **D:** token-wise nonlinearity and channel mixing. | Correct. |

## University of Padua, February 2023 NLP final, Problem 2

[Exam](https://stem.elearning.unipd.it/pluginfile.php/737219/mod_resource/content/1/20230220.pdf)

Condition: perform the first eight character-level BPE merges, including an end-of-word marker. At each iteration, show the adjacent-pair frequencies, the chosen merge, and the updated frequencies. The given word counts are `hug: 10`, `pug: 5`, `pun: 12`, `bun: 4`, `hugs: 5`.

**My response:** I skipped the calculation. I believe I understand BPE merges well from prior practice, but this particular exercise was not attempted or verified.

## Follow-up questions I proposed

The selected exams did not probe some architectural motivations I wanted to test. These are **my own question ideas**, not exam questions or answers:

1. Why use LayerNorm at all?
2. Why scale attention scores by `1/√d_k`?
3. Why use LayerNorm rather than BatchNorm in a Transformer?
