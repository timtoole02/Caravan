# Caravan 🐪

> **A zero-config, edge-native peer-to-peer cooperative inference engine for running large LLMs across a caravan of resource-constrained devices.**

---

Caravan is a lightweight, pure-Rust distributed inference engine built on top of the high-performance [NanoCamelid](https://github.com/timtoole02/NanoCamelid) runtime. It partitions transformer model layers across multiple machines on a local network, allowing smaller, memory-limited devices (like a swarm of Raspberry Pis or old laptops) to cooperatively execute inference on models that would otherwise trigger Out-Of-Memory (OOM) failures.

```
                     [ User Input / Client Node ]
                            │
                            │ (Send prompt tokens)
                            ▼
              [ Node 0 (e.g., Layers 0-11) ]  ◄── Pi 5
                            │
                            │ (8KB Activation Tensor)
                            ▼
              [ Node 1 (e.g., Layers 12-23) ] ◄── Pi 4
                            │
                            │ (8KB Activation Tensor)
                            ▼
              [ Node 2 (e.g., Layers 24-31) ] ◄── Final Sampler
                            │
                            │ (Return sampled token ID)
                            ▼
                     [ Output Token ]
```

---

## ⚡ The Edge-Swarm Advantage

* **Dynamic Layer Partitioning**: Partition model layers sequentially. If Node 0 runs layers 0–11 and Node 1 runs layers 12–23, each node only loads the weight parameters and allocates Key-Value (KV) cache memory for its assigned slice, completely bypassing single-device RAM bottlenecks.
* **Micro-Latency Network Passing**: During the forward pass, nodes transmit only the intermediate activation vector (the hidden state matrix, typically **~8KB of data**) to the next hop. Network transit takes **less than 0.1 milliseconds** on local Gigabit Ethernet or Wi-Fi, leaving network overhead virtually invisible.
* **Embedded Futuristic Web Console**: Caravan features a built-in, zero-dependency async HTTP server. Booting a client automatically hosts a high-end browser-based visualizer showing real-time compute-vs-network metrics, live chat, and an animated SVG topology mapping token flow particles.
* **Zero Dependencies, Zero Python**: A single compiled binary written in pure Rust. No heavy runtime frameworks, virtual environments, dynamic linking hassles, or complex GPU drivers.

---

## 🎨 Swarm Web Console

When you launch Caravan in client mode, it starts a lightweight HTTP server on port `7733`. Open **`http://localhost:7733`** to access:
* **Swarm Topology Visualizer**: A responsive, live diagram displaying active edge nodes, their layer splits, and IP addresses. Pulsing glow rings and animated particle streams light up the connections to track activation network transfers in real-time.
* **Split Compute charts**: Real-time telemetry gauges displaying Global throughput (tokens/second), Time-to-First-Token (TTFT), and raw Compute vs. Network transit delays using Chart.js.
* **Integrated Terminal Chat**: Send queries to the swarm directly in your browser, viewing streaming chat tokens in real-time.

---

## 🚀 Easy-Start Recipe (Local Simulation)

You don't need multiple physical machines to experience Caravan. You can simulate a fully functioning 3-node cooperative swarm right on your laptop!

### 1. Build the Swarm
Ensure you have the Rust toolchain installed, then clone and compile:
```bash
git clone https://github.com/timtoole02/Caravan.git
cd Caravan
cargo build --release
```

### 2. Launch the Swarm Nodes
Open **four separate terminal windows** and execute the following:

* **Terminal 1: Node 2 (Final Sampler - Layers 24..31)**:
  ```bash
  ./target/release/caravan --port 8002 --layers 24-31 --model /path/to/model.gguf
  ```

* **Terminal 2: Node 1 (Intermediate Worker - Layers 12..23)**:
  ```bash
  ./target/release/caravan --port 8001 --layers 12-23 --next-node 127.0.0.1:8002 --model /path/to/model.gguf
  ```

* **Terminal 3: Node 0 (Swarm Entry - Layers 0..11)**:
  ```bash
  ./target/release/caravan --port 8000 --layers 0-11 --next-node 127.0.0.1:8001 --model /path/to/model.gguf
  ```

* **Terminal 4: Swarm Client Chat & Dashboard**:
  ```bash
  ./target/release/caravan client --target 127.0.0.1:8000 --model /path/to/model.gguf
  ```

---

## 🛠️ Command-Line Interface Manual

```text
Usage: caravan [OPTIONS] [COMMAND]

Commands:
  client  Start interactive client chat driving the swarm
  help    Print this message or the help of the given subcommand(s)

Options:
      --port <PORT>              Port to listen on (for worker mode)
      --layers <LAYERS>          Contiguous range of layers to run (e.g., 0-15)
      --next-node <NEXT_NODE>    Address of the next pipeline node (e.g., 127.0.0.1:8001)
      --model <MODEL>            Path to GGUF model file
      --temp <TEMP>              Sampling temperature [default: 0.0]
      --max-tokens <MAX_TOKENS>  Maximum output token budget [default: 128]
  -h, --help                     Print help
```

---

## 🛜 Physical Deployment (e.g., Raspberry Pi Swarm)

To run a physical swarm across multiple local devices:
1. Copy the compiled `caravan` binary to all participating machines.
2. Ensure they are on the same local network and ports `8000-8002` are open.
3. Start the worker processes on their respective devices using their host IP addresses:
   * **Node 1 (Pi 4 - `192.168.1.101`)**:
     ```bash
     ./caravan --port 8001 --layers 16-31 --model /path/to/model.gguf
     ```
   * **Node 0 (Pi 5 - `192.168.1.100`)**:
     ```bash
     ./caravan --port 8000 --layers 0-15 --next-node 192.168.1.101:8001 --model /path/to/model.gguf
     ```
4. Run the Client CLI from any device (even a laptop/phone with low RAM loading only the tokenizer):
   ```bash
   ./caravan client --target 192.168.1.100:8000 --model /path/to/tokenizer.gguf
   ```
5. Open **`http://localhost:7733`** on the client machine to watch your physical edge swarm pulse and generate!

---

## 📄 License

Caravan is licensed under the MIT License. See [LICENSE](LICENSE) for details.
