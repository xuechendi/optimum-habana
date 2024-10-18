bs=64

TQDM_DISABLE=1 python run_generation.py \
	--model_name_or_path meta-llama/Llama-3.1-8B-Instruct  \
	--attn_softmax_bf16 \
	--use_hpu_graphs \
	--trim_logits \
	--warmup 2 \
	--use_kv_cache \
	--bucket_size=${bs} \
	--bucket_internal \
	--max_new_tokens 1024 \
	--max_input_tokens 1024 \
	--bf16 \
	--batch_size ${bs}  \
	--use_flash_attention \
	--output_dir generated_text_bf16_bs${bs}_1024_1024_fwd_lat.txt \
	--flash_attention_recompute 2>&1 | tee bf16_bs${bs}_1024_1024_fwd_lat.log
#--limit_hpu_graphs \
