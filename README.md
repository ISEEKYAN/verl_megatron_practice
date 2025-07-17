# verl_megatron_practice
(best/better) practices of megatron on veRL and tuning guide

This repo will contain extensive practices of megatron on veRL for reference of better utilization of GPUs. And also provide a guide for tuning veRL from aspect of megatron/veRL pipeline/rollout. And it will point out some optimization directions for veRL.


## Critical Optimization Options
### 1. Dynamic Batch Size
[see this PR](https://github.com/volcengine/verl/pull/1617). 

With dynamic batch size, we make micro batchs across time/GPUs to be balanced. It is recommended to enable it anyway.
And make the `max_token_len_per_gpu` as large as possible. The only limitation is the memory of GPUs.

[balanced dynamic batching](https://github.com/volcengine/verl/pull/2452) makes dynamic batching more balanced across DP and PP(1f1b). Reducing the bubbles.

```bash
use_dynamic_bsz=True
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
```


### 2. fused output kernel
[see this PR](https://github.com/volcengine/verl/pull/2210). This PR use fused kernel to eliminate the logits memory during training.
Before the fused kernel, we have an optimization to avoid recovering logits memory during training. 

[see this PR](https://github.com/volcengine/verl/pull/1629). 
These optimizations would reduce the peak memory usage for about 3-10GB, providing more room for other optimizations(Parallelism, batch size, etc.).

Recommend to enable it anyway.
```bash
actor_rollout_ref.model.use_fused_kernels=True \
```

### 3. offload
Offload parameters/optimizer/gradients to CPU. 
It is recommended to enable it. Unless the training batch size is too small.
```bash
actor_rollout_ref.actor.megatron.param_offload=${offload} \
actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
actor_rollout_ref.actor.megatron.grad_offload=${offload} \
actor_rollout_ref.ref.megatron.param_offload=${offload} \
```

### 4. DP/TP/PP/CP/EP/ETP parallelism
It is challenging to tune the parallelism for different models and different hardwares. The most usual limitation is the OOM of GPUs. 

We will try to give a better reference for it. Now please see other materials of pretraining on megatron, such as [Nemo's reference](https://github.com/NVIDIA/NeMo/blob/main/scripts/performance/recommended_model_configs/model_configs_h100.csv).

For memory estimation, please use my [estimator](https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator)

### 5. Other megatron optimizations
#### recompute
```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
```

more recompute granularity to be added.
#### 1f1b overlap
to be added.




### 6. inference engine specific optimization
a simple principle is to use the least parallelism and the largest `gpu_memory_utilization` as possible, unless OOM.

more details to be added.


### 7. Async training
to be added.


## Megatron Performance Optimization Options

Refer to [megatron_optim_options](./megatron_optim_options.md) to find more details of arguments adjustment to achieve better proformance.

## Practices

All benchmarks were conducted using Docker image `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview` and code from the `mcore_fused` branch of `https://github.com/ISEEKYAN/verl.git` (a veRL 0.4.0 derivative with custom features). Performance data reflects the 3rd training step.

-   **DAPO MATH with dense 7B model**
    -   **Note**: Requires modifying `max_position_embeddings` to `32768` in the `Qwen2.5-Math-7B`'s `config.json`.
    -   **Results**: [dapo7b.csv](./dapo7b.csv)
    -   **Scripts**: [DAPO_7B/](./DAPO_7B/)

-   **DAPO MATH with MoE 30B model**
    -   **Results**: [dapo_moe30b.csv](./dapo_moe30b.csv)
    -   **Scripts**: [DAPO_MOE30B/](./DAPO_MOE30B/)

-   **DAPO MATH with MoE 235B model**
    -   **Results**: [dapo_moe235b.csv](./dapo_moe235b.csv)
    -   **Scripts**: [DAPO_MOE235B/](./DAPO_MOE235B/)

-   **GRPO with qwen2.5vl-7B**
    -   **Results**: [grpo_qwen2_5vl.csv](./grpo_qwen2_5vl.csv)




