use std::path::Path;
use nanocamelid::model::{LlamaModelConfig, LlamaWeights};
use nanocamelid::inference::{
    LlamaKvCache, LlamaWorkspace, LlamaBatchWorkspace, LlamaRuntimeOptions,
    rms_norm, quantize_f32_to_q8_0, matmul_quantized, apply_rope, apply_attention_heads,
    AttentionInput, fused_silu_mul, matmul_f32, rms_norm_batch, quantize_f32_to_q8_0_batch,
    matmul_quantized_batch, add_bias, add_bias_batch, add_residual_batch, sample_logits,
    BatchMatmulShape,
};
use nanocamelid::q8::Q8_BLOCK_SIZE;

pub struct SwarmPipelineExecutor {
    pub config: LlamaModelConfig,
    pub weights: LlamaWeights,
    pub cache: LlamaKvCache,
    pub ws: LlamaWorkspace,
    pub batch_ws: LlamaBatchWorkspace,
    pub options: LlamaRuntimeOptions,
}

impl SwarmPipelineExecutor {
    pub fn new(
        model_path: &Path,
        q8_selector: nanocamelid::q8::Q8DotKernelSelector,
    ) -> Result<Self, String> {
        let gguf = nanocamelid::gguf::read_file(model_path)
            .map_err(|e| format!("failed to read GGUF: {e}"))?;

        let config = LlamaModelConfig::from_gguf(&gguf)
            .map_err(|e| format!("failed to parse config: {e}"))?;

        let weights = LlamaWeights::load(model_path, &config, &gguf)
            .map_err(|e| format!("failed to load weights: {e}"))?;

        let cache = LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let ws = LlamaWorkspace::new(&config);
        let batch_ws = LlamaBatchWorkspace::new(&config, 32); // Max batch 32 for swarm prefill

        let rope_scaling = nanocamelid::inference::RopeScaling::default(); // default
        let options = LlamaRuntimeOptions {
            q8_selector,
            compute_logits: true,
            rope_scaling,
        };

        Ok(Self {
            config,
            weights,
            cache,
            ws,
            batch_ws,
            options,
        })
    }

    /// Execute a range of layers for a single token decode pass
    pub fn run_decode_range(
        &mut self,
        start_layer: usize,
        end_layer: usize,
        token_id: Option<u32>,
        input_activations: Option<&[f32]>,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        // Step 1: Input Setup (Embedding lookup or received activation tensor)
        if start_layer == 0 {
            let Some(tid) = token_id else {
                return Err("Node 0 requires a token_id input".to_string());
            };
            let emb_start = tid as usize * self.config.embedding_length;
            self.ws.hidden.copy_from_slice(
                &self.weights.token_embeddings[emb_start..emb_start + self.config.embedding_length],
            );
        } else {
            let Some(activations) = input_activations else {
                return Err(format!("Intermediate layer range starting at {start_layer} requires input_activations"));
            };
            self.ws.hidden.copy_from_slice(activations);
        }

        // Step 2: Run layers
        for layer_idx in start_layer..=end_layer {
            let layer = &self.weights.layers[layer_idx];

            // Save residual
            self.ws.residual.copy_from_slice(&self.ws.hidden);

            // RMSNorm
            rms_norm(
                &mut self.ws.norm_x,
                &self.ws.hidden,
                &layer.attention_norm,
                self.config.rms_norm_epsilon,
            );

            // Quantize
            quantize_f32_to_q8_0(&self.ws.norm_x, &mut self.ws.x_i8, &mut self.ws.x_scales);

            // Projections (Q, K, V)
            matmul_quantized(
                &mut self.ws.q,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.wq,
                self.config.embedding_length,
                self.config.embedding_length,
                self.options.q8_selector,
            );
            if let Some(bias) = &layer.wq_bias {
                add_bias(&mut self.ws.q, bias);
            }
            matmul_quantized(
                &mut self.ws.k,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.wk,
                self.config.kv_width,
                self.config.embedding_length,
                self.options.q8_selector,
            );
            if let Some(bias) = &layer.wk_bias {
                add_bias(&mut self.ws.k, bias);
            }
            matmul_quantized(
                &mut self.ws.v,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.wav,
                self.config.kv_width,
                self.config.embedding_length,
                self.options.q8_selector,
            );
            if let Some(bias) = &layer.wav_bias {
                add_bias(&mut self.ws.v, bias);
            }

            // Apply RoPE
            apply_rope(
                &mut self.ws.q,
                pos,
                self.config.attention_head_count,
                self.config.head_dim,
                self.config.rope_dimension_count,
                self.config.rope_freq_base,
                self.options.rope_scaling,
            );
            apply_rope(
                &mut self.ws.k,
                pos,
                self.config.attention_head_count_kv,
                self.config.head_dim,
                self.config.rope_dimension_count,
                self.config.rope_freq_base,
                self.options.rope_scaling,
            );

            // Store KV Cache
            self.cache.store_kv(layer_idx, pos, &self.ws.k, &self.ws.v);

            // Retrieve K, V caches
            let k_cache = self.cache.get_k_cache(layer_idx);
            let v_cache = self.cache.get_v_cache(layer_idx);

            let scale = 1.0 / (self.config.head_dim as f32).sqrt();
            self.ws.attn_output.fill(0.0);
            apply_attention_heads(
                &mut self.ws.attn_output,
                &mut self.ws.attn_scores,
                AttentionInput {
                    q: &self.ws.q,
                    k_cache,
                    v_cache,
                    pos,
                    head_count: self.config.attention_head_count,
                    kv_head_count: self.config.attention_head_count_kv,
                    head_dim: self.config.head_dim,
                    cache_kv_width: self.cache.kv_width,
                    context_length: self.config.context_length,
                    scale,
                },
                false, // default to non-parallel attention heads in P2P single-token
            );

            // Projection O (wo)
            quantize_f32_to_q8_0(&self.ws.attn_output, &mut self.ws.x_i8, &mut self.ws.x_scales);
            matmul_quantized(
                &mut self.ws.hidden,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.wo,
                self.config.embedding_length,
                self.config.embedding_length,
                self.options.q8_selector,
            );

            // Residual addition
            for i in 0..self.config.embedding_length {
                self.ws.hidden[i] += self.ws.residual[i];
            }

            // --- FFN ---
            self.ws.residual.copy_from_slice(&self.ws.hidden);
            rms_norm(
                &mut self.ws.norm_x,
                &self.ws.hidden,
                &layer.ffn_norm,
                self.config.rms_norm_epsilon,
            );
            quantize_f32_to_q8_0(&self.ws.norm_x, &mut self.ws.x_i8, &mut self.ws.x_scales);

            matmul_quantized(
                &mut self.ws.ffn_gate,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.w1,
                self.config.feed_forward_length,
                self.config.embedding_length,
                self.options.q8_selector,
            );
            matmul_quantized(
                &mut self.ws.ffn_up,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.w3,
                self.config.feed_forward_length,
                self.config.embedding_length,
                self.options.q8_selector,
            );

            fused_silu_mul(&mut self.ws.ffn_gate_up, &self.ws.ffn_gate, &self.ws.ffn_up);
            quantize_f32_to_q8_0(&self.ws.ffn_gate_up, &mut self.ws.x_i8, &mut self.ws.x_scales);

            matmul_quantized(
                &mut self.ws.hidden,
                &self.ws.x_i8,
                &self.ws.x_scales,
                &layer.w2,
                self.config.embedding_length,
                self.config.feed_forward_length,
                self.options.q8_selector,
            );

            for i in 0..self.config.embedding_length {
                self.ws.hidden[i] += self.ws.residual[i];
            }
        }

        // Step 3: Final Norm and Logits Projection (Final Node only)
        if end_layer == self.config.block_count - 1 {
            rms_norm(
                &mut self.ws.norm_x,
                &self.ws.hidden,
                &self.weights.output_norm,
                self.config.rms_norm_epsilon,
            );
            if let Some(out_proj) = &self.weights.output_projection {
                quantize_f32_to_q8_0(&self.ws.norm_x, &mut self.ws.x_i8, &mut self.ws.x_scales);
                matmul_quantized(
                    &mut self.ws.logits,
                    &self.ws.x_i8,
                    &self.ws.x_scales,
                    out_proj,
                    self.config.vocab_size,
                    self.config.embedding_length,
                    self.options.q8_selector,
                );
            } else {
                matmul_f32(
                    &mut self.ws.logits,
                    &self.ws.norm_x,
                    &self.weights.token_embeddings,
                    self.config.vocab_size,
                    self.config.embedding_length,
                );
            }
            // Return the logits vector
            return Ok(self.ws.logits.clone());
        }

        // Return current hidden state to pass to the next node
        Ok(self.ws.hidden.clone())
    }

    /// Execute a range of layers for a batch prefill pass
    pub fn run_prefill_range(
        &mut self,
        start_layer: usize,
        end_layer: usize,
        token_ids: Option<&[u32]>,
        input_activations: Option<&[f32]>,
        start_pos: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let hidden_len = batch_size * self.config.embedding_length;

        // Step 1: Input Setup (Embedding lookup or received activation tensor)
        if start_layer == 0 {
            let Some(tids) = token_ids else {
                return Err("Node 0 requires token_ids input during prefill".to_string());
            };
            for (token_idx, &token_id) in tids.iter().enumerate() {
                let emb_start = token_id as usize * self.config.embedding_length;
                let hidden_start = token_idx * self.config.embedding_length;
                self.batch_ws.hidden[hidden_start..hidden_start + self.config.embedding_length]
                    .copy_from_slice(
                        &self.weights.token_embeddings
                            [emb_start..emb_start + self.config.embedding_length],
                    );
            }
        } else {
            let Some(activations) = input_activations else {
                return Err(format!("Intermediate layer range starting at {start_layer} requires input_activations"));
            };
            self.batch_ws.hidden[..hidden_len].copy_from_slice(&activations[..hidden_len]);
        }

        // Step 2: Run layers
        for layer_idx in start_layer..=end_layer {
            let layer = &self.weights.layers[layer_idx];
            let kv_len = batch_size * self.config.kv_width;

            self.batch_ws.residual[..hidden_len].copy_from_slice(&self.batch_ws.hidden[..hidden_len]);
            rms_norm_batch(
                &mut self.batch_ws.norm_x[..hidden_len],
                &self.batch_ws.hidden[..hidden_len],
                &layer.attention_norm,
                self.config.rms_norm_epsilon,
                batch_size,
                self.config.embedding_length,
            );
            quantize_f32_to_q8_0_batch(
                &self.batch_ws.norm_x[..hidden_len],
                &mut self.batch_ws.x_i8[..hidden_len],
                &mut self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                batch_size,
                self.config.embedding_length,
            );

            let attention_shape = BatchMatmulShape {
                batch_size,
                rows: self.config.embedding_length,
                cols: self.config.embedding_length,
            };
            matmul_quantized_batch(
                &mut self.batch_ws.q[..hidden_len],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.wq,
                attention_shape,
                self.options.q8_selector,
            );
            add_bias_batch(
                &mut self.batch_ws.q[..hidden_len],
                &layer.wq_bias,
                batch_size,
                self.config.embedding_length,
            );

            let kv_shape = BatchMatmulShape {
                batch_size,
                rows: self.config.kv_width,
                cols: self.config.embedding_length,
            };
            matmul_quantized_batch(
                &mut self.batch_ws.k[..kv_len],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.wk,
                kv_shape,
                self.options.q8_selector,
            );
            add_bias_batch(
                &mut self.batch_ws.k[..kv_len],
                &layer.wk_bias,
                batch_size,
                self.config.kv_width,
            );
            matmul_quantized_batch(
                &mut self.batch_ws.v[..kv_len],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.wav,
                kv_shape,
                self.options.q8_selector,
            );
            add_bias_batch(
                &mut self.batch_ws.v[..kv_len],
                &layer.wav_bias,
                batch_size,
                self.config.kv_width,
            );

            // Apply RoPE and store to KV cache
            for token_idx in 0..batch_size {
                let pos = start_pos + token_idx;
                let q_start = token_idx * self.config.embedding_length;
                let kv_start = token_idx * self.config.kv_width;
                apply_rope(
                    &mut self.batch_ws.q[q_start..q_start + self.config.embedding_length],
                    pos,
                    self.config.attention_head_count,
                    self.config.head_dim,
                    self.config.rope_dimension_count,
                    self.config.rope_freq_base,
                    self.options.rope_scaling,
                );
                apply_rope(
                    &mut self.batch_ws.k[kv_start..kv_start + self.config.kv_width],
                    pos,
                    self.config.attention_head_count_kv,
                    self.config.head_dim,
                    self.config.rope_dimension_count,
                    self.config.rope_freq_base,
                    self.options.rope_scaling,
                );
                self.cache.store_kv(
                    layer_idx,
                    pos,
                    &self.batch_ws.k[kv_start..kv_start + self.config.kv_width],
                    &self.batch_ws.v[kv_start..kv_start + self.config.kv_width],
                );
            }

            let k_cache = self.cache.get_k_cache(layer_idx);
            let v_cache = self.cache.get_v_cache(layer_idx);
            let scale = 1.0 / (self.config.head_dim as f32).sqrt();
            self.batch_ws.attn_output[..hidden_len].fill(0.0);

            for token_idx in 0..batch_size {
                let pos = start_pos + token_idx;
                let q_token_start = token_idx * self.config.embedding_length;
                let out_token_start = token_idx * self.config.embedding_length;
                let scores_start = token_idx * self.config.context_length;
                let scores = &mut self.batch_ws.attn_scores[scores_start..scores_start + self.config.context_length];

                apply_attention_heads(
                    &mut self.batch_ws.attn_output[out_token_start..out_token_start + self.config.embedding_length],
                    scores,
                    AttentionInput {
                        q: &self.batch_ws.q[q_token_start..q_token_start + self.config.embedding_length],
                        k_cache,
                        v_cache,
                        pos,
                        head_count: self.config.attention_head_count,
                        kv_head_count: self.config.attention_head_count_kv,
                        head_dim: self.config.head_dim,
                        cache_kv_width: self.cache.kv_width,
                        context_length: self.config.context_length,
                        scale,
                    },
                    false, // default to non-parallel
                );
            }

            quantize_f32_to_q8_0_batch(
                &self.batch_ws.attn_output[..hidden_len],
                &mut self.batch_ws.x_i8[..hidden_len],
                &mut self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                batch_size,
                self.config.embedding_length,
            );
            matmul_quantized_batch(
                &mut self.batch_ws.hidden[..hidden_len],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.wo,
                attention_shape,
                self.options.q8_selector,
            );
            add_residual_batch(&mut self.batch_ws.hidden[..hidden_len], &self.batch_ws.residual[..hidden_len]);

            // FFN
            self.batch_ws.residual[..hidden_len].copy_from_slice(&self.batch_ws.hidden[..hidden_len]);
            rms_norm_batch(
                &mut self.batch_ws.norm_x[..hidden_len],
                &self.batch_ws.hidden[..hidden_len],
                &layer.ffn_norm,
                self.config.rms_norm_epsilon,
                batch_size,
                self.config.embedding_length,
            );
            quantize_f32_to_q8_0_batch(
                &self.batch_ws.norm_x[..hidden_len],
                &mut self.batch_ws.x_i8[..hidden_len],
                &mut self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                batch_size,
                self.config.embedding_length,
            );

            let ffn_shape = BatchMatmulShape {
                batch_size,
                rows: self.config.feed_forward_length,
                cols: self.config.embedding_length,
            };
            matmul_quantized_batch(
                &mut self.batch_ws.ffn_gate[..batch_size * self.config.feed_forward_length],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.w1,
                ffn_shape,
                self.options.q8_selector,
            );
            matmul_quantized_batch(
                &mut self.batch_ws.ffn_up[..batch_size * self.config.feed_forward_length],
                &self.batch_ws.x_i8[..hidden_len],
                &self.batch_ws.x_scales[..batch_size * (self.config.embedding_length / Q8_BLOCK_SIZE)],
                &layer.w3,
                ffn_shape,
                self.options.q8_selector,
            );

            for token_idx in 0..batch_size {
                let start = token_idx * self.config.feed_forward_length;
                let end = start + self.config.feed_forward_length;
                fused_silu_mul(
                    &mut self.batch_ws.ffn_gate_up[start..end],
                    &self.batch_ws.ffn_gate[start..end],
                    &self.batch_ws.ffn_up[start..end],
                );
            }

            quantize_f32_to_q8_0_batch(
                &self.batch_ws.ffn_gate_up[..batch_size * self.config.feed_forward_length],
                &mut self.batch_ws.x_i8[..batch_size * self.config.feed_forward_length],
                &mut self.batch_ws.x_scales[..batch_size * (self.config.feed_forward_length / Q8_BLOCK_SIZE)],
                batch_size,
                self.config.feed_forward_length,
            );

            let down_shape = BatchMatmulShape {
                batch_size,
                rows: self.config.embedding_length,
                cols: self.config.feed_forward_length,
            };
            matmul_quantized_batch(
                &mut self.batch_ws.hidden[..hidden_len],
                &self.batch_ws.x_i8[..batch_size * self.config.feed_forward_length],
                &self.batch_ws.x_scales[..batch_size * (self.config.feed_forward_length / Q8_BLOCK_SIZE)],
                &layer.w2,
                down_shape,
                self.options.q8_selector,
            );
            add_residual_batch(&mut self.batch_ws.hidden[..hidden_len], &self.batch_ws.residual[..hidden_len]);
        }

        // Final Node step in prefill
        if end_layer == self.config.block_count - 1 {
            // For batch prefill, we usually only need to sample the *last* token to kickstart decode.
            // Let's compute logits for the very last token in the batch.
            let last_token_idx = batch_size - 1;
            let hidden_offset = last_token_idx * self.config.embedding_length;
            let last_hidden = &self.batch_ws.hidden[hidden_offset..hidden_offset + self.config.embedding_length];

            rms_norm(
                &mut self.ws.norm_x,
                last_hidden,
                &self.weights.output_norm,
                self.config.rms_norm_epsilon,
            );

            if let Some(out_proj) = &self.weights.output_projection {
                quantize_f32_to_q8_0(&self.ws.norm_x, &mut self.ws.x_i8, &mut self.ws.x_scales);
                matmul_quantized(
                    &mut self.ws.logits,
                    &self.ws.x_i8,
                    &self.ws.x_scales,
                    out_proj,
                    self.config.vocab_size,
                    self.config.embedding_length,
                    self.options.q8_selector,
                );
            } else {
                matmul_f32(
                    &mut self.ws.logits,
                    &self.ws.norm_x,
                    &self.weights.token_embeddings,
                    self.config.vocab_size,
                    self.config.embedding_length,
                );
            }
            // Return logits for the last token so that the system can sample it
            return Ok(self.ws.logits.clone());
        }

        // Return hidden activations of all tokens in batch
        Ok(self.batch_ws.hidden[..hidden_len].to_vec())
    }

    /// Helper to sample from logits
    pub fn sample(&self, logits: &[f32], temp: f32) -> u32 {
        sample_logits(logits, temp) as u32
    }
}
