# Optimization through Megatron Performance Improvement Options

## Introduction

To achieve high efficiency and MFU, we should:

- Overlap Communication/Computation to reduce computaion bubbles
- Make full use of Memory to achieve maximum Computation Intensity, but no OOM
    - Balance between model states memory and activation memory

## Communication/Computation Overlap

- tp_comm_overlap
- overlap_grad_reduce (DDP)
- overlap_param_gather
- overlap_param_gather_with_optimizer_step
- moe_shared_expert_overlap

## Operators Fusion

Most operators default enable fusion in `megatron.training.arguments`, but not enabled in configurtion files.

- masked_softmax_fusion
- bias_gelu_fusion
- bias_swiglu_fusion
- bias_dropout_fusion
- apply_rope_fusion
- cross_entropy_loss_fusion (implemented by verl)
    - cross_entropy_fusion_impl: `te`
- gradient_accumulation_fusion

## MoE Optimization

### Options

- moe_shared_expert_overlap
- moe_group_gemm
- **moe_enable_deepep**: Provide most performance gain with inter-node experts communication
- moe_permute_fusion
- moe_token_dispatcher_type: flex

## Fine-grained Recomputation

Through fine-grained recomputation, with the nearly unchanged performance, training will shows activation memory decrease.

- recompute_granularity: selective
- recompute_modules: [core_attn, mlp, moe, moe_act, layernorm, mla_up_proj], the latter three use output-discatding checkpointing
- recompute_method: uniform
- recompute_num_layers: 1, together with uniform recompute method, force recompute in every layers

## Parallelism

Best Practice (determined in order):

### Tensor Model Parallel(TP)

TP reduces the model weights and the computation volume, but introducing frequent communication. TP size is often less than `NPROCS_PER_NODE` (GPU number within 1 node) so the communication is often within a machine.

**TLDR: The best TP size is where computation just cover the communication so that allows fully overlap. since TP communication is in huge ammount, TP size is as small as possible.**

### Context Parallel (SP/Ulysses/CP/Hierarchical)

Sequence Parallel works together with TP and its **intra-node** communication can be overlapped by computation in Megatron, verl opens it in default.

Ulysses CP works effectively within 1 node, with 4 non-overlapable all-to-all communication in forward and backward, which harms MFU. Ulysses CP requires UCP size larger than KV heads, which suffers limitations in GQA models, especially DeepSeek-V3 (MLA only have 1 kv head).

Context Parallel decreases computation by $O(CP^2)$, and introduces constant communication volume in total: KV of all machines transmitted in CP times, with $\frac{KV}{CP}$ every time. Context Parallel is suitable for **intra/inter-node** communication. This is still a linear function to select suitable CP like TP before, to make sure CP calculation can overlap inter-node communication.

Hierarchical Context Parallel allows to use Ulysses CP + Context Parallel, suitable for extremely over-long sequences.

**TLDR: SP default enables, USP$=\min{\frac{NPROCS\_PER\_NODE}{TP}, kv\_head}$, CP computation overlappable by its inter-node communication.**

### Expert Model Parallel(EP) and Expert Tensor Parallel(ETP)

Since TP+CP mainly work for attention operator, EP and ETP work for MoE operator.

EP decreases MoE experts in one machine and is independent with TP in other layers. EP decrease calculation in a single machine, but introduces communication of combine and dispatch all-to-all communication, accelarated by Deep-EP.

Since models like DeepSeek-V3 has larger MoE layer compared with Attention, EP is the better way to reduce memory overhead. Considering Pipeline bubbles introduced by PP, EP is the best parallelization way in MoE models.

EP comp-comm overlap can only be enabled with overlapped 1F1B Pipeline, similar with DualPipe. This improves performance a lot because MoE dispatch and combine communication is more than forward pass of attention and MoE calculation.

ETP is not recommanded to use since its matrix-multiplication is smaller and more frequent than EP, which is inefficient.

**TLDR: ETP disable, EP works with overlapped 1F1B Pipeline + Deep-EP**

### Pipeline Model Parallel(PP), Virtual Pipeline Model Parallel(VPP) and Flexible Pipeline Layout

PP devides model by layer. Each PP rank owns $\frac{total\_layers}{PP}$ layers but devided into VPP size of incontinuous layer blocks.

Pipeline Parallelism bubble rate is $\frac{1}{v}\cdot \frac{p-1}{m}$([1]), so increasing micro-batch-size and VPP size helps improve performance. While micro-batch-size can increase activation memory pressure, here is a trade-off.

Flexible Pipeline Layout: Since models like qwen3-moe 235B or DeepSeek-V3 is more complicated with dense/moe layers or MTP, Pipeline Stages need better layout to achieve memory balance.

**TLDR: When using Pipeline Parallel, we need to increase micro-batch-size and VPP to handle pipeline bubbles. And Pipeline layout needs carefully adjustment to achieve balance**

#### Data Parallel

Data Parallel size is set automatically to $\frac{WORLD\_SIZE}{TP\times CP\times PP}$, determining iteration time but only affects distributed optimizer in mini-batch time.

## Extra

Good practice by NVIDIA: [Megatron-MoE-ModelZoo](https://github.com/yanring/Megatron-MoE-ModelZoo)

DeepSeek-V3 pretrain best pactice, by NVIDIA:

![dpsk-V3](./images/dpsk-V3-best-practice.png)

## Reference

[1] Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., ... & Zaharia, M. (2021, November). Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the international conference for high performance computing, networking, storage and analysis (pp. 1-15).