# CellTriage API Performance Optimization Report

**Date:** January 9, 2026

## Executive Summary

The CellTriage API worker configuration was optimized to improve ingress throughput and stability. Benchmark testing with 1000 PMIDs shows processing in ~6 minutes with the optimized configuration.

## Benchmark Results (1000 PMIDs)

| Metric | Value |
|--------|-------|
| Total PMIDs submitted | 1000 |
| Total time | 5.7 minutes |
| **Throughput** | **175 PMIDs/min** |

*Benchmark performed January 9, 2026 with optimized configuration (GPU inference).*

## Problem Statement

The ingress worker was experiencing:
1. No explicit concurrency limit (defaulted to all 16 cores)
2. No thread limits for CPU operations
3. Potential tokenizer deadlocks in forked workers

## Configuration Changes

### Before (docker-compose.dev.yml)

```yaml
worker-ingress:
  command: celery ... -Q ingress --hostname=ingress@%h
  # No concurrency limit, no thread limits

worker-infer:
  command: celery ... -Q infer --hostname=infer@%h --pool=solo
  environment:
    LOAD_CT_TAGGER: yes
    # No thread limits
```

### After (docker-compose.dev.yml)

```yaml
worker-ingress:
  command: celery ... -Q ingress --hostname=ingress@%h --concurrency=4 --prefetch-multiplier=1
  environment:
    OMP_NUM_THREADS: "4"
    MKL_NUM_THREADS: "4"
    TOKENIZERS_PARALLELISM: "false"

worker-infer:
  command: celery ... -Q infer --hostname=infer@%h --pool=solo
  environment:
    LOAD_CT_TAGGER: yes
    OMP_NUM_THREADS: "4"
    MKL_NUM_THREADS: "4"
    TOKENIZERS_PARALLELISM: "false"
```

## Key Optimizations

### 1. Ingress Worker Concurrency (--concurrency=4)
- **Problem:** Default concurrency used all 16 cores, excessive for I/O-bound work
- **Solution:** Limit to 4 concurrent SIBiLS fetch tasks
- **Impact:** More controlled resource usage, reduced memory pressure

### 2. Thread Limiting (OMP_NUM_THREADS=4)
- **Problem:** CPU operations could spawn unlimited threads
- **Solution:** Limit to 4 threads per worker
- **Impact:** Prevents CPU contention during text preprocessing

### 3. Prefetch Multiplier Reduced (--prefetch-multiplier=1)
- **Problem:** High prefetch causes bursty load patterns
- **Solution:** Prefetch 1 task per worker slot
- **Impact:** Smoother load distribution

### 4. Tokenizer Parallelism Disabled
- Standard practice for production Celery deployments
- Prevents potential deadlocks in forked workers

## Architecture Notes

CellTriage has a split worker architecture:

| Worker | Queue | Pool | Purpose |
|--------|-------|------|---------|
| worker-ingress | ingress | prefork (4) | SIBiLS fetch, text extraction |
| worker-infer | infer | solo | GPU inference (NVIDIA) |

The `--pool=solo` for inference is correct because:
- GPU inference is single-threaded
- CUDA context cannot be shared across forked processes
- Solo pool ensures sequential GPU task execution

## Server Specifications

- **CPU:** 16 cores
- **RAM:** 62 GB
- **GPU:** NVIDIA (for inference)
- **Model:** Custom CellTriage tagger
- **Framework:** PyTorch (GPU inference)

## Conclusion

The optimization improves stability and resource utilization. With GPU inference, CellTriage achieves ~175 PMIDs/min throughput (~10,500/hour), making it highly suitable for large-scale cell biology literature triage.
