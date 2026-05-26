# Caravan

A zero-config, edge-native peer-to-peer cooperative inference engine for running large LLMs across a caravan of resource-constrained devices.

Caravan is built on top of the high-performance, pure-Rust [NanoCamelid](https://github.com/timtoole02/NanoCamelid) runtime. It partitions transformer model layers across multiple machines on a local network, allowing smaller, memory-limited devices (like Raspberry Pis) to cooperatively execute inference.

## Key Features

- **Cooperative Layer Splitting**: Partition model layers across a pipeline of nodes (e.g. Node A runs layers 0–15, Node 1 runs layers 16–31).
- **Reduced Memory Footprint**: Each node only loads weights and allocates Key-Value (KV) cache pages for the layers it is assigned to, preventing OOM issues on low-RAM devices.
- **Negligible Network Latency**: Only intermediate activation vectors (typically ~8KB of data) are sent across the TCP streams, taking under 0.1ms to transmit on local networks.
- **No Python/C++ Build Steps**: Fully self-contained, pure Rust implementation.

## How It Works

```
                     [ User Input / Client Node ]
                            │
                            │ (Send prompt tokens)
                            ▼
              [ Node 0 (e.g., Layers 0-15) ]
                            │
                            │ (Activation Tensor)
                            ▼
              [ Node 1 (e.g., Layers 16-31) ]
                            │
                            │ (Activation Tensor)
                            ▼
              [ Node 2 (e.g., Layers 32-47) ]
                            │
                            │ (Sample token ID)
                            ▼
                     [ Return token ID ]
```

1. The **Client** accepts prompt text, tokenizes it, and sends the prompt tokens to **Node 0**.
2. **Node 0** initializes/reuses its local KV cache segment, performs the forward pass for layers `0..K_0`, and serializes the final activation matrix.
3. The activation matrix is passed over a fast TCP stream to **Node 1**.
4. **Node 1** executes layers `K_0+1..K_1` on the input, then passes the result to the next node.
5. The **Final Node** computes logits, samples the next token ID, and returns the token ID directly back to the Client to output in real-time.

## Quick Start

### Build
Ensure you have the Rust toolchain installed:
```bash
cargo build --release
```

### Run (Local Simulation)
To run a local 3-process simulation on your machine using a single GGUF model:

1. **Start the final node (Node 2 - layers 24..31)**:
   ```bash
   cargo run --release -- --port 8002 --layers 24-31 --model /path/to/model.gguf
   ```
2. **Start the intermediate node (Node 1 - layers 12..23)**:
   ```bash
   cargo run --release -- --port 8001 --layers 12-23 --next-node 127.0.0.1:8002 --model /path/to/model.gguf
   ```
3. **Start the starting node (Node 0 - layers 0..11)**:
   ```bash
   cargo run --release -- --port 8000 --layers 0-11 --next-node 127.0.0.1:8001 --model /path/to/model.gguf
   ```
4. **Launch the interactive client**:
   ```bash
   cargo run --release -- client --target 127.0.0.1:8000 --model /path/to/model.gguf
   ```
