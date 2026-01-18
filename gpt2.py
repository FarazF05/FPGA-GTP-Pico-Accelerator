import numpy as np
import time
from accelerator import (
    USE_FPGA,
    USE_FPGA_IN_LOOP,
    fpga_gemm_tile,
    dequantize_from_int32,
    verify_fpga_result,
    get_timing_log,
    reset_timing_log,
    _timing_log
)

TILE_M = 4 # how many rows of tokens we take
TILE_K = 4 # how many input features we take
TILE_N = 4 # how many output features we take

token_timing_history = []

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    """
    If USE_FPGA is True:
        - Sends a small tile to FPGA for computation
        - Verifies FPGA result against CPU reference
        - Logs detailed timing breakdown
    
    If USE_FPGA_IN_LOOP is also True:
        - Actually uses the FPGA result in the model output
        - NOTE: This only works correctly when TILE_K == x.shape[-1] (full inner dimension)
        - Otherwise FPGA computes a partial sum which cannot replace the CPU result
    """
    global _timing_log

    t_total_start = time.perf_counter()

    # main cpu computation (always done)
    t_cpu_start = time.perf_counter()
    y = x @ w + b
    t_cpu_end = time.perf_counter()

    if not USE_FPGA:
        _timing_log["cpu_remaining_s"] += t_cpu_end - t_cpu_start
        _timing_log["total_s"] += t_cpu_end - t_cpu_start
        return y
    
    # extract tile for FPGA 
    if x.ndim == 2:
        inner_dim = x.shape[1]
        A = x[:TILE_M, :TILE_K]
        B = w[:TILE_K, :TILE_N]
        bias_tile = b[:TILE_N] if b is not None else None
    elif x.ndim == 3:
        inner_dim = x.shape[2]
        A = x[0, :TILE_M, :TILE_K]
        B = w[:TILE_K, :TILE_N]
        bias_tile = b[:TILE_N] if b is not None else None    
    else:
        # fallback, just use CPU
        _timing_log["cpu_remaining_s"] += t_cpu_end - t_cpu_start
        _timing_log["total_s"] += time.perf_counter() - t_total_start
        return y
    
    # Check if we're computing a full tile or partial sum
    # FPGA computes A[:, :TILE_K] @ B[:TILE_K, :]
    # This is only equal to (x @ w)[:TILE_M, :TILE_N] if TILE_K == inner_dim
    is_full_tile = (TILE_K >= inner_dim)

    c_fpga, scale_A, scale_B, fpga_cycles = fpga_gemm_tile(A, B)
    
    matches, max_err = verify_fpga_result(A, B, c_fpga, scale_A, scale_B)
    if not matches:
        print(f"FPGA mismatch! Max error: {max_err}")
    
    # Dequantization and write-back (only if computing full tile AND flag is set)
    if USE_FPGA_IN_LOOP and is_full_tile:
        t_dequant_start = time.perf_counter()
        c_float = dequantize_from_int32(c_fpga, scale_A, scale_B)
        if bias_tile is not None:
            c_float = c_float + bias_tile
        t_dequant_end = time.perf_counter()
        _timing_log["dequantization_s"] += t_dequant_end - t_dequant_start
        
        # Write FPGA result back into output
        if x.ndim == 2:
            y[:TILE_M, :TILE_N] = c_float
        elif x.ndim == 3:
            y[0, :TILE_M, :TILE_N] = c_float
    elif USE_FPGA_IN_LOOP and not is_full_tile:
        # Still time dequantization for logging purposes, but don't write back
        t_dequant_start = time.perf_counter()
        c_float = dequantize_from_int32(c_fpga, scale_A, scale_B)
        t_dequant_end = time.perf_counter()
        _timing_log["dequantization_s"] += t_dequant_end - t_dequant_start
        # Note: Not writing back because FPGA computed partial sum, not full result
    
    # Record CPU time (the full matmul we did at the start)
    _timing_log["cpu_remaining_s"] += t_cpu_end - t_cpu_start
    
    t_total_end = time.perf_counter()
    _timing_log["total_s"] += t_total_end - t_total_start
    
    return y
    

def log_token_timing(token_id=None):
    """
    Call this after each token generation to save timing data.
    Returns the timing log for this token.
    """
    timing = get_timing_log()
    timing["token_id"] = token_id
    timing["timestamp"] = time.time()
    token_timing_history.append(timing)
    reset_timing_log()
    return timing


def get_timing_summary(timing=None):
    """Pretty print a timing breakdown."""
    if timing is None:
        timing = get_timing_log()
    
    print("\n" + "=" * 50)
    print("TIMING BREAKDOWN")
    print("=" * 50)
    print(f"  Quantization:      {timing['quantization_s']*1000:8.3f} ms")
    print(f"  UART Send:         {timing['uart_send_s']*1000:8.3f} ms")
    print(f"  UART Receive:      {timing['uart_recv_s']*1000:8.3f} ms")
    print(f"  Dequantization:    {timing['dequantization_s']*1000:8.3f} ms")
    print(f"  CPU Remaining:     {timing['cpu_remaining_s']*1000:8.3f} ms")
    print("-" * 50)
    print(f"  TOTAL:             {timing['total_s']*1000:8.3f} ms")
    print("-" * 50)
    print(f"  FPGA Cycles:       {timing['fpga_cycles']:>8d}")
    if timing.get('scale_A', 0) > 0:
        print(f"  Scale A:           {timing['scale_A']:8.3f}")
        print(f"  Scale B:           {timing['scale_B']:8.3f}")
    print("=" * 50 + "\n")


def save_timing_history_csv(filepath="timing_log.csv"):
    """Save all token timing history to CSV for analysis."""
    import csv
    
    if not token_timing_history:
        print("No timing data to save.")
        return
    
    fieldnames = [
        "token_id", "timestamp", "total_s",
        "quantization_s", "uart_send_s", "uart_recv_s",
        "dequantization_s", "cpu_remaining_s",
        "fpga_cycles", "scale_A", "scale_B"
    ]
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in token_timing_history:
            writer.writerow(row)
    
    print(f"Saved {len(token_timing_history)} timing records to {filepath}")


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
