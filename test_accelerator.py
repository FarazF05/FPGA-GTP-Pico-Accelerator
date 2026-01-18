"""
Test suite for FPGA accelerator integration.

This tests the full pipeline with a mock FPGA that does the same math
as the real hardware would do. Verifies:
1. Quantization correctness
2. Protocol correctness  
3. Timing logging
4. FPGA-in-loop mode
"""

import numpy as np
import time
import io

# Configure before importing
import accelerator
accelerator.USE_FPGA = True
accelerator.USE_FPGA_IN_LOOP = True

import gpt2
gpt2.USE_FPGA = True
gpt2.USE_FPGA_IN_LOOP = True


class MockFPGA:
    """
    Simulates FPGA behavior for testing.
    Performs the same int8 x int8 -> int32 math that real FPGA does.
    """
    
    def __init__(self):
        self.input_buffer = io.BytesIO()
        self.output_buffer = io.BytesIO()
        self.cycle_count = 0
        self.last_A = None
        self.last_B = None
    
    def write(self, data):
        """Receive data from CPU (simulates UART RX on FPGA side)."""
        self.input_buffer.write(data)
        self._process_if_complete()
    
    def flush(self):
        """No-op for mock."""
        pass
    
    def read(self, n_bytes):
        """Send data to CPU (simulates UART TX on FPGA side)."""
        # Simulate some FPGA compute latency
        time.sleep(0.001)  # 1ms simulated compute time
        return self.output_buffer.read(n_bytes)
    
    def reset_input_buffer(self):
        self.input_buffer = io.BytesIO()
    
    def reset_output_buffer(self):
        self.output_buffer = io.BytesIO()
    
    def _process_if_complete(self):
        """Check if we have a complete request and process it."""
        data = self.input_buffer.getvalue()
        
        # Protocol: 1 byte start + 16 bytes A + 16 bytes B = 33 bytes
        if len(data) < 33:
            return
        
        # Verify start byte
        if data[0] != 0xA5:
            print(f"ERROR: Invalid start byte {hex(data[0])}")
            return
        
        # Extract matrices
        A_bytes = data[1:17]
        B_bytes = data[17:33]
        
        A_int8 = np.frombuffer(A_bytes, dtype=np.int8).reshape(4, 4)
        B_int8 = np.frombuffer(B_bytes, dtype=np.int8).reshape(4, 4)
        
        self.last_A = A_int8.copy()
        self.last_B = B_int8.copy()
        
        # Do the math (exactly what FPGA does)
        C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
        
        # Simulate cycle count (4x4 systolic takes ~7-8 cycles to fill + drain)
        self.cycle_count = 16  # Just a placeholder
        
        # Prepare response: 64 bytes result + 4 bytes cycle count
        self.output_buffer = io.BytesIO()
        self.output_buffer.write(C_int32.astype('<i4').tobytes())
        self.output_buffer.write(np.array([self.cycle_count], dtype='<u4').tobytes())
        self.output_buffer.seek(0)
        
        # Clear input buffer
        self.input_buffer = io.BytesIO()


def test_quantization():
    """Test that quantization preserves information reasonably."""
    print("\n" + "="*60)
    print("TEST: Quantization")
    print("="*60)
    
    # Test various input ranges
    test_cases = [
        np.array([[0.5, -0.5], [0.25, -0.25]]),  # Small values
        np.array([[100.0, -100.0], [50.0, -50.0]]),  # Large values
        np.array([[0.0, 0.0], [0.0, 0.0]]),  # Zeros
        np.random.randn(4, 4).astype(np.float32),  # Random
    ]
    
    for i, x in enumerate(test_cases):
        x_int8, scale = accelerator.quantize_to_int8(x)
        x_recovered = x_int8.astype(np.float64) / scale
        max_err = np.max(np.abs(x - x_recovered))
        
        print(f"  Case {i+1}: max_abs={np.max(np.abs(x)):.4f}, "
              f"scale={scale:.4f}, recovery_error={max_err:.6f}")
        
        # Should recover within ~1/127 of max value
        if np.max(np.abs(x)) > 1e-9:
            assert max_err < np.max(np.abs(x)) / 64, f"Recovery error too large: {max_err}"
    
    print("  PASSED")


def test_protocol():
    """Test that the protocol produces correct byte sequences."""
    print("\n" + "="*60)
    print("TEST: Protocol")
    print("="*60)
    
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=np.int8)
    
    B = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.int8)  # Identity
    
    packet = accelerator.pack_request(A, B)
    
    print(f"  Packet length: {len(packet)} bytes (expected: 33)")
    print(f"  Start byte: {hex(packet[0])} (expected: 0xa5)")
    
    assert len(packet) == 33
    assert packet[0] == 0xA5
    
    # Verify A is correctly packed
    A_recovered = np.frombuffer(packet[1:17], dtype=np.int8).reshape(4, 4)
    assert np.array_equal(A, A_recovered), "A not correctly packed"
    
    # Verify B is correctly packed
    B_recovered = np.frombuffer(packet[17:33], dtype=np.int8).reshape(4, 4)
    assert np.array_equal(B, B_recovered), "B not correctly packed"
    
    print("  PASSED")


def test_fpga_mock():
    """Test end-to-end with mock FPGA."""
    print("\n" + "="*60)
    print("TEST: Mock FPGA Integration")
    print("="*60)
    
    # Install mock
    mock = MockFPGA()
    accelerator._ser = mock
    accelerator.get_serial = lambda: mock
    
    # Reset timing
    accelerator.reset_timing_log()
    
    # Create test data
    A = np.random.randn(4, 4).astype(np.float32)
    B = np.random.randn(4, 4).astype(np.float32)
    
    # Run through FPGA path
    C_fpga, scale_A, scale_B, cycles = accelerator.fpga_gemm_tile(A, B)
    
    # Compute reference
    C_ref, ref_scale_A, ref_scale_B = accelerator.cpu_reference_int8(A, B)
    
    # Verify
    matches, max_err = accelerator.verify_fpga_result(A, B, C_fpga, scale_A, scale_B)
    
    print(f"  Scales: A={scale_A:.3f}, B={scale_B:.3f}")
    print(f"  FPGA cycles: {cycles}")
    print(f"  Max error: {max_err}")
    print(f"  Matches: {matches}")
    
    # Check timing was logged
    timing = accelerator.get_timing_log()
    print(f"  Timing logged: quant={timing['quantization_s']*1000:.3f}ms, "
          f"send={timing['uart_send_s']*1000:.3f}ms, "
          f"recv={timing['uart_recv_s']*1000:.3f}ms")
    
    assert matches, f"FPGA result doesn't match reference! Error: {max_err}"
    assert timing['quantization_s'] > 0, "Quantization time not logged"
    
    print("  PASSED")


def test_fpga_in_loop():
    """Test that FPGA correctly computes partial sums and write-back works for full tiles."""
    print("\n" + "="*60)
    print("TEST: FPGA In Loop")
    print("="*60)
    
    # Install mock
    mock = MockFPGA()
    accelerator._ser = mock
    accelerator.get_serial = lambda: mock
    
    # Reset timing
    accelerator.reset_timing_log()
    
    # Test 1: Partial sum case (TILE_K < inner_dim)
    # FPGA computes A[:4,:4] @ B[:4,:4] which is NOT the same as (x@w)[:4,:4]
    print("  Case 1: Partial sum (TILE_K=4, inner_dim=16)")
    
    x = np.random.randn(8, 16).astype(np.float32)
    w = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)
    
    # CPU reference for the partial computation (what FPGA actually does)
    A = x[:4, :4]
    B = w[:4, :4]
    partial_cpu = A @ B  # This is the partial sum
    
    # Verify FPGA computes this partial sum correctly
    c_fpga, scale_A, scale_B, _ = accelerator.fpga_gemm_tile(A, B)
    c_dequant = accelerator.dequantize_from_int32(c_fpga, scale_A, scale_B)
    
    partial_error = np.max(np.abs(partial_cpu - c_dequant))
    print(f"    Partial sum error: {partial_error:.6f}")
    assert partial_error < 0.1, f"Partial sum error too large: {partial_error}"
    
    # Test 2: Full tile case (TILE_K == inner_dim)
    # When inner dimension matches tile size, FPGA computes the full result
    print("  Case 2: Full tile (TILE_K=4, inner_dim=4)")
    
    accelerator.reset_timing_log()
    
    x_small = np.random.randn(8, 4).astype(np.float32)  # inner_dim = 4 = TILE_K
    w_small = np.random.randn(4, 8).astype(np.float32)
    b_small = np.random.randn(8).astype(np.float32)
    
    y_cpu = x_small @ w_small + b_small
    y_hybrid = gpt2.linear(x_small, w_small, b_small)
    
    # Now the tile SHOULD match (within quantization error)
    tile_diff = np.max(np.abs(y_cpu[:4, :4] - y_hybrid[:4, :4]))
    relative_error = tile_diff / (np.max(np.abs(y_cpu[:4, :4])) + 1e-9)
    
    print(f"    Tile max diff: {tile_diff:.6f}")
    print(f"    Relative error: {relative_error*100:.2f}%")
    
    # Non-tile region should be identical
    y_cpu_rest = y_cpu.copy()
    y_hybrid_rest = y_hybrid.copy()
    y_cpu_rest[:4, :4] = 0
    y_hybrid_rest[:4, :4] = 0
    rest_matches = np.allclose(y_cpu_rest, y_hybrid_rest)
    
    print(f"    Non-tile region matches: {rest_matches}")
    
    assert rest_matches, "Non-tile region was modified!"
    assert relative_error < 0.05, f"Full tile error too large: {relative_error*100:.2f}%"
    
    print("  PASSED")


def test_timing_logging():
    """Test timing history and CSV export."""
    print("\n" + "="*60)
    print("TEST: Timing Logging")
    print("="*60)
    
    # Install mock
    mock = MockFPGA()
    accelerator._ser = mock
    accelerator.get_serial = lambda: mock
    
    # Clear history
    gpt2.token_timing_history.clear()
    
    # Simulate a few tokens
    for i in range(3):
        accelerator.reset_timing_log()
        
        x = np.random.randn(8, 16).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)
        
        _ = gpt2.linear(x, w, b)
        
        gpt2.log_token_timing(token_id=1000 + i)
    
    history = gpt2.get_timing_history()
    print(f"  Logged {len(history)} tokens")
    
    assert len(history) == 3, f"Expected 3 timing records, got {len(history)}"
    
    # Check structure
    for record in history:
        assert 'total_s' in record
        assert 'quantization_s' in record
        assert 'token_id' in record
        print(f"    Token {record['token_id']}: total={record['total_s']*1000:.3f}ms")
    
    # Test CSV export
    gpt2.save_timing_history_csv("/tmp/test_timing.csv")
    
    # Verify CSV was created
    import csv
    with open("/tmp/test_timing.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        print(f"  CSV export verified: {len(rows)} rows")
    
    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# FPGA ACCELERATOR TEST SUITE")
    print("#"*60)
    
    test_quantization()
    test_protocol()
    test_fpga_mock()
    test_fpga_in_loop()
    test_timing_logging()
    
    print("\n" + "#"*60)
    print("# ALL TESTS PASSED")
    print("#"*60 + "\n")


if __name__ == "__main__":
    run_all_tests()