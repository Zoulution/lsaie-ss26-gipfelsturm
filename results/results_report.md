## Experiment 1: Tensor parallelism sweep

### 760M, 4 nodes, seq_len=4096
Purpose:
Test whether tensor parallelism improves throughput for the 760M model.

Setup:
- Model: 760M
- Nodes: 4
- GPUs: 16
- Seq length: 4096
- Global batch size: 256
- Micro batch size: 4
- PP=1, CP=1
- Attention backend: auto
- Precision: bf16

Results:
| TP | DP | Mean tok/s/GPU | Mean iter time | Final loss |
|---:|---:|---:|---:|---:|
| 1 | 16 | 31,230 | 2.16 s | 6.698 |
| 2 | 8 | 18,083 | 3.79 s | 6.784 |
| 4 | 4 | 12,197 | 5.43 s | 6.726 |

Conclusion:
TP=2 gives ~42% lower throughput than TP=1.
TP=4 gives ~61% lower throughput than TP=1.

For 760M, tensor parallelism significantly reduces throughput. TP=1 is best because the model fits on each GPU and TP communication overhead dominates. TP should likely be reserved for larger models or memory-limited settings.

### 1.5B, 4 nodes, seq_len=4096
| TP | DP | tok/s/GPU | total tok/s | iter ms | TFLOPs/GPU | Relative throughput |
| -: | -: | --------: | ----------: | ------: | ---------: | ------------------: |
|  1 | 16 |    34,632 |     554,118 |    1893 |        352 |               1.00× |
|  2 |  8 |    20,010 |     320,160 |    3335 |        203 |               0.58× |

same pattern: TP=2 gives only ~58% of TP=1 throughput.

### comparison
| Model | Layers | Hidden | tok/s/GPU | TFLOPs/GPU | iter ms |
| ----- | -----: | -----: | --------: | ---------: | ------: |
| 760M  |     24 |   1536 |    31,230 |        154 |    2158 |
| 1.5B  |     48 |   1600 |    34,632 |        352 |    1893 |

The 760M model is too small to fully saturate the GPUs.
The 1.5B model has better compute/communication ratio.

##
| Nodes | GPUs | DP |    tok/s/GPU |   Total tok/s |   Iter ms | TFLOPs/GPU |
| ----: | ---: | -: | -----------: | ------------: | --------: | ---------: |
|     1 |    4 |  4 |     40,145.8 |       160,583 |    6573.1 |      198.2 |
|     2 |    8 |  8 | **49,102.3** |       392,818 |    2756.9 |  **242.4** |
|     4 |   16 | 16 |     31,230.4 |       499,686 |    2158.0 |      154.2 |
|     8 |   32 | 32 |     35,883.2 | **1,148,263** | **946.8** |      177.1 |
Total throughput:
1n → 2n → 4n → 8n improves overall

Per-GPU throughput:
best at 2n, worse at 4n and 8n

Potential reason:
Main reason: fixed global batch causes less work per GPU
When DP increases, the number of gradient accumulation steps decreases. At 4 and 8 nodes, each GPU does fewer microsteps per optimizer step. That means there is less compute available to hide fixed overheads

Second reason: DP all-reduce gets more expensive