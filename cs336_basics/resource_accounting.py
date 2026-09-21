"""
Problem (transformer_accounting): Transformer LM resource accounting (5 points)

(a) Consider a GPT-2 XL-sized model using our assignment architecture, which
has the following configuration:

    vocab_size:     50,257
    context_length: 1,024
    num_layers:     48
    d_model:        1,600
    num_heads:      25
    d_ff:           4,288
                    (the nearest multiple of 64 to (8 / 3) × 1,600)

Suppose we constructed our model using this configuration. How many trainable
parameters would our model have? Assuming each parameter is represented using
single-precision floating point, how much memory is required to just load this
model?

Deliverable: A one-to-two sentence response.

Answer(a): 
We have: embeddings -> layers * num_layers -> final_norm -> lm_head
embeddings: [vocab_size, d_model]
layer:
    attention_prenorm: g vector, [d_model]
    mha: q, k, v, o projs, [suppose d_q=d_k=d_v=d_model // num_heads], 4 * [d_model, d_model]
    ffn_prenorm: [d_model]
    ffn: w1: [d_model, d_ff], w2: [d_ff, d_model], w3: [d_model, d_ff]
final_norm: [d_model]
lm_head: [d_model, vocab_size]


(b) Identify the matrix multiplies required to complete a forward pass of our
GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in
total? Assume that our input sequence has context_length tokens.

Deliverable: A list of matrix multiplies (with descriptions), and the total
number of FLOPs required.

inside layer:
    inside_silu = einsum(self.W_silu, x, 'd_ff d_model, ... d_model -> ... d_ff')
    near_silu = einsum(self.W_inner, x, 'd_ff d_model, ... d_model -> ... d_ff')
    ffn_result = einsum(self.W_outer, inside, 'd_model d_ff, ... d_ff -> ... d_model')
    
    queries = einsum(x, self.W_q, "... d_model, hd d_model -> ... hd")
    keys = einsum(x, self.W_k, "... d_model, hd d_model -> ... hd")
    values = einsum(x, self.W_v, "... d_model, hd d_model -> ... hd")
    presoftmax_numerator = einsum(queries, keys, "... q_seq_len d_k, ... k_seq_len d_k -> ... q_seq_len k_seq_len")
    sdpa_result = einsum(softmaxed, values, "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v")
    mha_result = einsum(self.W_o, sdpa_result_unslised, "d_model hd, ... hd -> ... d_model")

all of it n times..

+ lm_head: 2 x batch_size x seq_len x d_model x vocab_size

so, if we suppose x has size [batch_size, seq_len, d_model](after embeddings) each layer has:
3 * FFN_FLOPs = 2 x batch_size x seq_len x d_ff x d_model
4 * proj_FLOPS = 2 x batch_size x seq_len x d_model x d_model
1 * kv_presoftmax_FLOPS = 2 * batch_size x seq_len x seq_len x d_model
1 * sftmx_values_FLOPS = 2 * batch_size x num_heads x seq_len x seq_len x (d_model // num_heads)

Total number of FLOPS: 3516769894400

(c) Based on your analysis above, which parts of the model require the most
FLOPs?

Deliverable: A one-to-two sentence response.

If we take the sum of all layers - FFN(~2023GFLOPs), 
if we're talking bout single matrix multiplication - lm head(~164GFLOPs)

(d) Repeat your analysis with:

    GPT-2 small:
        num_layers: 12
        d_model:    768
        num_heads:  12

    GPT-2 medium:
        num_layers: 24
        d_model:    1,024
        num_heads:  16

    GPT-2 large:
        num_layers: 36
        d_model:    1,280
        num_heads:  20

As the model size increases, which parts of the Transformer LM take up
proportionally more or less of the total FLOPs?

Deliverable: For each model, provide a breakdown of model components and its
associated FLOPs (as a proportion of the total FLOPs required for a forward
pass). In addition, provide a one-to-two sentence description of how varying
the model size changes the proportional FLOPs of each component.

As model size grows, FFN component and attention projections grows, while LM becomes smaller and smaller(27 -> 12 -> 7 %)

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total
FLOPs for one forward pass change? How does the relative contribution of FLOPs
of the model components change?

Deliverable: A one-to-two sentence response.

Total GFLOPs: 3516.7698944 -> 133577.7296384 (~39x more, while context became only 16x more)
most "fat" operation: FFN(57.5%) -> 24.23%
our new "kings": QK(30.86%) AND (softmax @ V)(30.86%) - both from 4.57%

I guess that's because of quadratic complexity of the attention over context len, we can see it throught formulaes
"""

vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288
d_head = d_model // num_heads # (64)


def calc_size(sizes: list[int]):
    if len(sizes) == 1:
        return sizes[0]
    elif len(sizes) == 2:
        return sizes[0] * sizes[1]
    else:
        print("...hmm")

from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    d_model: int
    num_heads: int
    vocab_size: int = 50257
    context_length: int | None = None
    d_ff: int | None = None


def calc_and_print_trainable_params(config: ModelConfig, print_extra: bool = False):
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    d_model = config.d_model
    d_ff = config.d_ff if config.d_ff is not None else (8 * d_model // 3)

    num_params_embeddings = calc_size([vocab_size, d_model])
    num_params_norm = calc_size([d_model])
    num_params_mha = calc_size([d_model, d_model]) * 4
    num_params_ffn = calc_size([d_ff, d_model]) * 3
    num_params_lm_head = calc_size([d_model, vocab_size])

    if print_extra:
        print("embeddings has ", num_params_embeddings, " params")
        print("norm has ", num_params_norm, " params")
        print("mha has ", num_params_mha // 4, " params")
        print("ffn has ", num_params_ffn // 3, " params")
        print("lm head has ", num_params_lm_head, " params")

    total_params = num_params_embeddings + (2 * num_layers + 1) * num_params_norm + num_layers * (num_params_mha + num_params_ffn) + num_params_lm_head
    if print_extra:
        print(total_params, " = ", total_params / 10 ** 9, "billions params")

    return total_params


num_params_embeddings = calc_size([vocab_size, d_model])
num_params_norm = calc_size([d_model])
num_params_mha = calc_size([d_model, d_model]) * 4
num_params_ffn = calc_size([d_ff, d_model]) * 3
num_params_lm_head = calc_size([d_model, vocab_size])

# layers = [[d_model] * 2 + [d_model, d_model] * 4 + [d_ff, d_model] * 3] * num_layers
print("embeddings has ", num_params_embeddings, " params")
print("norm has ", num_params_norm, " params")
print("mha has ", num_params_mha // 4, " params")
print("ffn has ", num_params_ffn // 3, " params")
print("lm head has ", num_params_lm_head, " params")

answer_a = num_params_embeddings + (2 * num_layers + 1) * num_params_norm + num_layers * (num_params_mha + num_params_ffn) + num_params_lm_head
print(answer_a, " = ", answer_a / 10 ** 9, "billions params")
print("gpt2-XL requires ", answer_a * 4 / 10 ** 9, "GB for fp32(4 bytes)")

# from cs336_basics.transformer import TransformerLM

# transformer = TransformerLM(
#     d_model,
#     num_heads,
#     d_ff,
#     1000,
#     vocab_size,
#     context_length,
#     num_layers
# )

# total_params = 0
# for name, p in transformer.named_parameters():
#     if p.requires_grad == True:
#         total_params += p.numel()
#         # print(name, " has ", p.numel(), " params")

# print(total_params, " = ", total_params // 10 ** 9, "billions params")

# ============================================
batch_size = 1

FFN_FLOPs = (2 * batch_size * context_length * d_ff * d_model) * 3
proj_FLOPs = (2 * batch_size * context_length * d_model * d_model) * 4
qk_presoftmax_FLOPs = 2 * batch_size * context_length * context_length * d_model
sftmx_values_FLOPs = 2 * batch_size * num_heads * context_length * context_length * (d_model // num_heads)
lm_head_FLOPs = 2 * batch_size * context_length * d_model * vocab_size
total_FLOPs = num_layers * (FFN_FLOPs + proj_FLOPs + qk_presoftmax_FLOPs + sftmx_values_FLOPs) + lm_head_FLOPs
print("FFN_FLOPs ", FFN_FLOPs / 10 ** 9 * num_layers)
print("proj_FLOPs ", proj_FLOPs / 10 ** 9 * num_layers)
print("kv_presoftmax_FLOPs ", qk_presoftmax_FLOPs / 10 ** 9 * num_layers)
print("sftmx_values_FLOPs ", sftmx_values_FLOPs / 10 ** 9 * num_layers)
print("lm_head_FLOPs ", lm_head_FLOPs / 10 ** 9)
print("TOTAL number of GFLOPs(bs = 1, seq_len = context_len): ", total_FLOPs / 10 ** 9)
print("~" * 66)

def calc_flops(config: ModelConfig, print_extra: bool = False):
    batch_size = 1
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    d_model = config.d_model
    context_length = config.context_length
    d_ff = config.d_ff if config.d_ff is not None else (8 * d_model // 3)

    FFN_FLOPs = (2 * batch_size * context_length * d_ff * d_model) * 3
    proj_FLOPs = (2 * batch_size * context_length * d_model * d_model) * 4
    qk_presoftmax_FLOPs = 2 * batch_size * context_length * context_length * d_model
    sftmx_values_FLOPs = 2 * batch_size * num_heads * context_length * context_length * (d_model // num_heads)
    lm_head_FLOPs = 2 * batch_size * context_length * d_model * vocab_size
    total_FLOPs = num_layers * (FFN_FLOPs + proj_FLOPs + qk_presoftmax_FLOPs + sftmx_values_FLOPs) + lm_head_FLOPs
    if print_extra:
        print("FFN_FLOPs percent: ", num_layers * FFN_FLOPs / total_FLOPs * 100, "%")
        print("proj_FLOPs percent: ", num_layers * proj_FLOPs / total_FLOPs * 100, "%")
        print("qk_presoftmax_FLOPs percent: ", num_layers * qk_presoftmax_FLOPs / total_FLOPs * 100, "%")
        print("sftmx_values_FLOPs percent: ", num_layers * sftmx_values_FLOPs / total_FLOPs * 100, "%")
        print("lm_head_FLOPs percent: ", lm_head_FLOPs / total_FLOPs * 100, "%")
        print("TOTAL number of GFLOPs(bs = 1, seq_len = context_len): ", total_FLOPs / 10 ** 9)

    return total_FLOPs


gpt2_small_config = ModelConfig(
    name="gpt2-small",
    num_layers=12,
    d_model=768,
    num_heads=12,
    context_length=1024
)
total_params_small = calc_and_print_trainable_params(gpt2_small_config)
print("total params for gpt2-small: ", total_params_small / 10 ** 9, " billion params")
total_flops_small = calc_flops(gpt2_small_config, print_extra=True)
print("~" * 66)

gpt2_medium_config = ModelConfig(
    name="gpt2-medium",
    num_layers=24,
    d_model=1024,
    num_heads=16,
    context_length=1024
)
total_params_medium = calc_and_print_trainable_params(gpt2_medium_config)
print("total params for gpt2-medium: ", total_params_medium / 10 ** 9, " billion params")
total_flops_medium = calc_flops(gpt2_medium_config, print_extra=True)
print("~" * 66)

gpt2_large_config = ModelConfig(
    name="gpt2-large",
    num_layers=36,
    d_model=1280,
    num_heads=20,
    context_length=1024
)
total_params_large = calc_and_print_trainable_params(gpt2_large_config)
print("total params for gpt2-large: ", total_params_large / 10 ** 9, " billion params")
total_flops_large = calc_flops(gpt2_large_config, print_extra=True)
print("~" * 66)

gpt2_xl_config = ModelConfig(
    name="gpt2-xl",
    vocab_size = 50257,
    context_length = 1024,
    num_layers = 48,
    d_model = 1600,
    num_heads = 25,
    d_ff = 4288
)
total_flops_xl = calc_flops(gpt2_xl_config, print_extra=True)
print("~" * 66)

gpt2_xl_config_long_context = ModelConfig(
    name="gpt2-xl-long-context",
    vocab_size = 50257,
    context_length = 16384,
    num_layers = 48,
    d_model = 1600,
    num_heads = 25,
    d_ff = 4288
)
total_flops_xl_long_context = calc_flops(gpt2_xl_config_long_context, print_extra=True)