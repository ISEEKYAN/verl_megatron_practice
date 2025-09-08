set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

rollout_mode="sync"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

wandb login YOUR_WANDB_KEY

gsm8k_train_path=YOURDATA_PATH/gsm8k/train.parquet
gsm8k_test_path=YOURDATA_PATH/gsm8k/test.parquet
math_train_path=YOURDATA_PATH/math/train.parquet
math_test_path=YOURDATA_PATH/math/test.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path']"
current_date=`date +%Y%m%d-%H%M`
# Nsight profiling configuration
# PROFILE_STEPS="[2,3,4]" # or [] or null
# PROFILE_RANKS_ALL=True # or True
# DISCRETE=True  # or True

PROFILE=${PROFILE:-0}
if [ $PROFILE -eq 1 ]; then
    echo "Profiling is enabled"
    PROFILE_STEPS="[2]"
    DISCRETE=True
    PROFILE_RANKS_ALL=False
    TOTAL_TRAINING_STEPS=10
    PROFILE_RANKS=[0,1,2,3,4,5,6,7]
else
    PROFILE_STEPS=null
    TOTAL_TRAINING_STEPS=null
    DISCRETE=True
    PROFILE_RANKS_ALL=False
    PROFILE_RANKS=null
fi


use_dynamic_bsz=True
max_prompt_length=2048
max_response_length=8192
enable_overlong_buffer=True
offload=True
gen_tp=1
train_tp=2
train_pp=1
CP=1
USE_FUSED_KERNELS=True

PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=512 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) )) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 10)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 10)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.enforce_eager=True  \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_qwen7b_megatron' \
    trainer.experiment_name='i2ko8k_bs512n16_tp4pp1cp1_lbmbs4_gentp1_offload_0.6' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    2>&1 | tee qwen2.5_7B_${current_date}.log 2>&1 &

    # trainer.profile_steps=$PROFILE_STEPS \
    # actor_rollout_ref.profiler.ranks=$PROFILE_RANKS \
    # actor_rollout_ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    # actor_rollout_ref.profiler.discrete=$DISCRETE \


