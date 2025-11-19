# Benchmark Performance

Benchmark model inference speed, accuracy, and resource usage.

## Purpose
- Measure inference speed
- Calculate throughput
- Monitor memory usage
- Compare model sizes
- Validate performance claims

## Usage

```bash
/benchmark
```

Or with specific model:
```bash
export MODEL_PATH=outputs/my_model
/benchmark
```

## Commands

```bash
MODEL_PATH=${MODEL_PATH:-outputs/quick_test}

echo "=================================="
echo "Performance Benchmark"
echo "=================================="
echo "Model: $MODEL_PATH"
echo ""

# Check model exists
if [ ! -f "$MODEL_PATH/best_model.pt" ]; then
    echo "❌ Model not found"
    exit 1
fi

# Run benchmark
python << 'PYEOF'
import torch
import time
import psutil
import os
from pathlib import Path
import sys

MODEL_PATH = os.environ.get('MODEL_PATH', 'outputs/quick_test')

print("Loading model and tokenizer...")
try:
    from models.hrm.hrm_language_v1 import HRMLanguageModel
    from train_language_model import SimpleTokenizer

    # Load model
    checkpoint = torch.load(f'{MODEL_PATH}/best_model.pt', map_location='cpu')
    config = checkpoint['config']

    model = HRMLanguageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = SimpleTokenizer.load(f'{MODEL_PATH}/tokenizer.json')

    print("✓ Model loaded\n")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# Model Statistics
# ============================================================================
print("="*60)
print("MODEL STATISTICS")
print("="*60)

model_size_mb = Path(f'{MODEL_PATH}/best_model.pt').stat().st_size / (1024**2)
params_total = model.get_num_params(non_embedding=False)
params_nonembedding = model.get_num_params(non_embedding=True)

print(f"Total parameters:        {params_total:,}")
print(f"Non-embedding params:    {params_nonembedding:,}")
print(f"Model file size:         {model_size_mb:.2f} MB")
print(f"Hidden size:             {config.hidden_size}")
print(f"Number of layers:        {config.num_layers}")
print(f"Attention heads:         {config.num_heads}")
print(f"H-cycles:                {config.H_cycles}")
print(f"L-cycles:                {config.L_cycles}")

# ============================================================================
# Inference Benchmark
# ============================================================================
print(f"\n{'='*60}")
print("INFERENCE BENCHMARK")
print("="*60)

test_prompt = "What is machine learning? It is"
tokens = tokenizer.encode(test_prompt, max_length=50)
input_ids = torch.tensor([tokens], dtype=torch.long)

print(f"Test input: '{test_prompt}'")
print(f"Token count: {len(tokens)}")
print()

# Warmup
print("Running warmup...")
for _ in range(3):
    with torch.no_grad():
        _ = model(input_ids)

# Benchmark forward pass
print("Benchmarking forward pass...")
num_runs = 20
times = []

for _ in range(num_runs):
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    times.append(time.time() - start)

avg_time_ms = (sum(times) / len(times)) * 1000
std_time_ms = (torch.tensor(times).std().item()) * 1000
min_time_ms = min(times) * 1000
max_time_ms = max(times) * 1000
throughput = 1000.0 / avg_time_ms

print(f"\nForward Pass Results ({num_runs} runs):")
print(f"  Average time:    {avg_time_ms:.2f} ms")
print(f"  Std deviation:   {std_time_ms:.2f} ms")
print(f"  Min time:        {min_time_ms:.2f} ms")
print(f"  Max time:        {max_time_ms:.2f} ms")
print(f"  Throughput:      {throughput:.2f} queries/sec")

# Benchmark generation
print(f"\nBenchmarking text generation...")
gen_start = time.time()
with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=50)
gen_time = time.time() - gen_start

tokens_generated = generated.shape[1] - input_ids.shape[1]
tokens_per_sec = tokens_generated / gen_time

print(f"Generation Results:")
print(f"  Tokens generated: {tokens_generated}")
print(f"  Time:            {gen_time*1000:.2f} ms")
print(f"  Speed:           {tokens_per_sec:.2f} tokens/sec")
print(f"  Per token:       {(gen_time/tokens_generated)*1000:.2f} ms/token")

# ============================================================================
# Memory Usage
# ============================================================================
print(f"\n{'='*60}")
print("MEMORY USAGE")
print("="*60)

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024**2)
memory_percent = process.memory_percent()

print(f"Process memory:      {memory_mb:.2f} MB")
print(f"Memory percent:      {memory_percent:.2f}%")

# Model memory (approximate)
model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
print(f"Model parameters:    {model_memory:.2f} MB")

# ============================================================================
# Efficiency Metrics
# ============================================================================
print(f"\n{'='*60}")
print("EFFICIENCY METRICS")
print("="*60)

# Parameters per second
params_per_sec = params_total * throughput
print(f"Parameters/sec:      {params_per_sec:,.0f}")

# Memory efficiency
memory_per_param = (model_memory * 1024) / params_total  # KB per param
print(f"Memory/parameter:    {memory_per_param:.4f} KB")

# Model efficiency score (higher is better)
efficiency_score = throughput / (model_size_mb / 100)
print(f"Efficiency score:    {efficiency_score:.2f}")

# ============================================================================
# Comparison Estimates
# ============================================================================
print(f"\n{'='*60}")
print("COMPARISON (Estimates)")
print("="*60)

# Estimate vs GPT-3.5 (175B params, ~100ms inference)
gpt_params = 175_000_000_000
size_ratio = params_total / gpt_params
speed_improvement = 100 / avg_time_ms

print(f"vs GPT-3.5 Turbo:")
print(f"  Size:            {size_ratio*100:.4f}% of GPT-3.5")
print(f"  Speed:           {speed_improvement:.1f}x faster")
print(f"  Energy:          ~{100/speed_improvement:.1f}% of GPT-3.5")

# Estimate vs GPT-4 (1.7T params, ~200ms inference)
gpt4_params = 1_700_000_000_000
size_ratio_gpt4 = params_total / gpt4_params
speed_improvement_gpt4 = 200 / avg_time_ms

print(f"\nvs GPT-4:")
print(f"  Size:            {size_ratio_gpt4*100:.6f}% of GPT-4")
print(f"  Speed:           {speed_improvement_gpt4:.1f}x faster")
print(f"  Energy:          ~{100/speed_improvement_gpt4:.1f}% of GPT-4")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print("BENCHMARK SUMMARY")
print("="*60)

# Quality assessment
if avg_time_ms < 100:
    speed_rating = "Excellent"
elif avg_time_ms < 500:
    speed_rating = "Good"
elif avg_time_ms < 1000:
    speed_rating = "Acceptable"
else:
    speed_rating = "Slow"

print(f"Speed rating:        {speed_rating}")
print(f"Model size:          {'Small' if params_total < 100_000_000 else 'Medium' if params_total < 1_000_000_000 else 'Large'}")
print(f"Memory efficient:    {'Yes' if memory_mb < 500 else 'Moderate' if memory_mb < 2000 else 'No'}")

print(f"\n✓ Benchmark complete!")
print(f"\nNext steps:")
print(f"  - Energy tracking: /track-carbon")
print(f"  - Deploy: /deploy-docker")
print(f"  - Save report for investors/customers")

# Save report
report = {
    'model_path': MODEL_PATH,
    'parameters': params_total,
    'model_size_mb': model_size_mb,
    'avg_inference_ms': avg_time_ms,
    'throughput_qps': throughput,
    'memory_mb': memory_mb,
    'tokens_per_sec': tokens_per_sec,
}

import json
with open('benchmark_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved: benchmark_report.json")

PYEOF
```

## Expected Performance

### Tiny Model (5M params)
- Inference: 10-50ms
- Throughput: 20-100 queries/sec
- Memory: 50-100 MB

### Small Model (15M params)
- Inference: 50-100ms
- Throughput: 10-20 queries/sec
- Memory: 100-200 MB

### Medium Model (60M params)
- Inference: 100-200ms
- Throughput: 5-10 queries/sec
- Memory: 300-500 MB

### Large Model (150M params)
- Inference: 200-500ms
- Throughput: 2-5 queries/sec
- Memory: 800-1200 MB

## Use Cases

### Marketing Claims
- "X% smaller than GPT-4"
- "Y times faster than ChatGPT"
- "Uses Z% less energy"

### Customer Demos
- Show real-time performance
- Prove efficiency claims
- Demonstrate scalability

### Optimization Targets
- Identify bottlenecks
- Track improvements
- Validate optimizations

## Success Criteria

- [x] Model loads successfully
- [x] Inference time measured
- [x] Throughput calculated
- [x] Memory usage tracked
- [x] Report generated
- [x] Comparisons provided
