import numpy as np
import time
import os
import serial

USE_FPGA = False
USE_FPGA_IN_LOOP = False

RESPONSE_SIZE = 68 # expected responce size of fpga output

START_BYTE = 0xA5
SER_PORT = os.environ.get("FPGA_PORT", "/dev/ttyUSB0")
BAUD = int(os.environ.get("FPGA_BAUD", "115200"))
TIMEOUT_S = 0.5

_ser = None


# Timing log for current token
_timing_log = {
    "quantization_s": 0.0,
    "uart_send_s": 0.0,
    "fpga_compute_s": 0.0,
    "uart_recv_s": 0.0,
    "dequantization_s": 0.0,
    "cpu_remaining_s": 0.0,
    "total_s": 0.0,
    "fpga_cycles": 0,
    "scale_A": 0.0,
    "scale_B": 0.0,
}


def get_timing_log():
    return _timing_log.copy()


def reset_timing_log():
    global _timing_log
    _timing_log["quantization_s"] = 0.0
    _timing_log["uart_send_s"] = 0.0
    _timing_log["fpga_compute_s"] = 0.0
    _timing_log["uart_recv_s"] = 0.0
    _timing_log["dequantization_s"] = 0.0
    _timing_log["cpu_remaining_s"] = 0.0
    _timing_log["total_s"] = 0.0
    _timing_log["fpga_cycles"] = 0
    _timing_log["scale_A"] = 0.0
    _timing_log["scale_B"] = 0.0


def quantize_to_int8(x):
    """
    Turns float numbers into small integers (int8)
    Used for the UART protocol which sends bytes (will send 1 byte per number)
    """
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-9:
        scale = 1.0
    else:
        scale = 127.0 / max_abs

    x_scaled = np.round(x * scale)
    x_clipped = np.clip(x_scaled, -128, 127)
    return x_clipped.astype(np.int8), scale


def dequantize_from_int32(c_int32, scale_A, scale_B):
    """
    Convert int32 accumulator back to float.
    Since we computed (A * scale_A) @ (B * scale_B) = C * (scale_A * scale_B)
    We divide by both scales to recover the original range.
    """
    return c_int32.astype(np.float64) / (scale_A * scale_B)


def pack_request(A, B):
    """
    Builds the exact bytes being sent to the UART
    
    :param A: 4x4 int8
    :param B: 4x4 int8
    """
    header = bytes([START_BYTE])
    payload = A.tobytes() + B.tobytes()
    return header + payload # 32 + 1 bytes in total


def get_serial():
    """
    Open the serial port once and reuse it
    """
    global _ser
    if _ser is None:
        _ser = serial.Serial(SER_PORT, BAUD, timeout=TIMEOUT_S)
        time.sleep(0.05) # some boards reset when port opens
        _ser.reset_input_buffer()
        _ser.reset_output_buffer()
    return _ser


def uart_send(packet_bytes):
    """
    Send bytes to FPGA over UART
    """
    ser = get_serial()
    ser.write(packet_bytes)
    ser.flush()


def uart_read_result():
    """
    Read 68 bytes from FPGA
    - 64 bytes: 16 x int32 result matrix
    - 4 bytes: uint32 cycle count
    Interpret them as 16 int32 numbers (4x4)
    """
    ser = get_serial()
    data = ser.read(RESPONSE_SIZE)
    if len(data) != RESPONSE_SIZE:
        raise TimeoutError(f"Expected {RESPONSE_SIZE} bytes but got {len(data)}")

    C_int32 = np.frombuffer(data[:64], dtype="<i4").reshape(4,4)
    cycle_count = np.frombuffer(data[64:68], dtype="u4")[0]
    return C_int32, cycle_count


def cpu_reference_int8(A, B):
    """
    Does the same math on CPU that FPGA is doing for comparison
    """
    A_int8, scale_A = quantize_to_int8(A)
    B_int8, scale_B = quantize_to_int8(B)
    C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
    return C_int32, scale_A, scale_B


def fpga_gemm_tile(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Quantizes float values to int8, sends to FPGA, reads int32 result.
    Returns (C_int32, scale_A, scale_B, fpga_cycles)
    
    If USE_FPGA is False or FPGA fails, falls back to CPU reference.
    """
    global _timing_log

    if not USE_FPGA:
        t0 = time.perf_counter()
        C_int32, scale_A, scale_B = cpu_reference_int8(A, B)
        t1 = time.perf_counter()
        _timing_log["quantization_s"] += t1 - t0
        return C_int32, scale_A, scale_B, 0

    # quantisation
    t_quant_start = time.perf_counter()
    A_int8, scale_A = quantize_to_int8(A)
    B_int8, scale_B = quantize_to_int8(B)
    packet = pack_request(A_int8, B_int8)
    t_quant_end = time.perf_counter()
    _timing_log["quantization_s"] += t_quant_end - t_quant_start
    _timing_log["scale_A"] = scale_A
    _timing_log["scale_B"] = scale_B
    

    try:
        ser = get_serial()
        ser.reset_input_buffer()

        # UART send
        t_send_start = time.perf_counter()
        uart_send(packet)
        t_send_end = time.perf_counter()
        _timing_log["uart_send_s"] += t_send_end - t_send_start

        # UART receive
        t_recv_start = time.perf_counter()
        C_int32, fpga_cycles = uart_read_result()
        t_recv_end = time.perf_counter()

        # total receive time includes FPGA compute + UART transfer
        # we can estimate FPGA compute time from cycle count if we know clock freq
        _timing_log["uart_recv_s"] += t_recv_end - t_recv_start
        _timing_log["fpga_cycles"] = fpga_cycles
        
        return C_int32, scale_A, scale_B, fpga_cycles
        
    except Exception as e:
        print(f"FPGA error: {e}, falling back to CPU")
        C_int32, scale_A, scale_B = cpu_reference_int8(A, B)
        return C_int32, scale_A, scale_B, 0


def verify_fpga_result(A, B, c_fpga, scale_A, scale_B):
    """
    Compare FPGA int32 result to CPU int32 reference computed using
    the exact same quantization scales.
    Returns (matches: bool, max_error: int)
    """
    A_q = np.clip(np.round(A * scale_A), -128, 127).astype(np.int8)
    B_q = np.clip(np.round(B * scale_B), -128, 127).astype(np.int8)

    c_cpu = A_q.astype(np.int32) @ B_q.astype(np.int32)

    diff = c_cpu - c_fpga
    max_err = int(np.max(np.abs(diff)))
    return (max_err == 0), max_err


