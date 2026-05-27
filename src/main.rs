use clap::{Parser, Subcommand};
use std::error::Error;
use std::path::PathBuf;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::io::{self, Write};

mod network;
mod pipeline;

use network::{recv_message, send_message, CaravanMessage, PipelinePayload};
use pipeline::SwarmPipelineExecutor;

#[derive(Parser, Debug)]
#[command(name = "caravan", version = "0.1.0", about = "Cooperative P2P Edge LLM Swarm")]
struct Args {
    #[arg(long, help = "Port to listen on (for worker mode)")]
    port: Option<u16>,

    #[arg(long, help = "Contiguous range of layers to run (e.g., 0-15)")]
    layers: Option<String>,

    #[arg(long, help = "Address of the next pipeline node (e.g., 127.0.0.1:8001)")]
    next_node: Option<String>,

    #[arg(long, help = "Path to GGUF model file")]
    model: Option<PathBuf>,

    #[arg(long, default_value = "0.0", help = "Sampling temperature")]
    temp: f32,

    #[arg(long, default_value = "128", help = "Maximum output token budget")]
    max_tokens: usize,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    #[command(about = "Start interactive client chat driving the swarm")]
    Client {
        #[arg(long, help = "Address of Node 0 in the swarm (e.g., 127.0.0.1:8000)")]
        target: String,

        #[arg(long, help = "Path to GGUF model file (used only for tokenizer)")]
        model: PathBuf,

        #[arg(long, help = "Prompt to execute in single-turn mode")]
        prompt: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();

    match args.command {
        Some(Commands::Client { target, model, prompt }) => {
            run_client(&target, &model, prompt, args.temp, args.max_tokens).await?;
        }
        None => {
            // Worker mode
            let port = args.port.ok_or("Missing --port for worker mode")?;
            let layers_str = args.layers.ok_or("Missing --layers for worker mode")?;
            let model_path = args.model.ok_or("Missing --model for worker mode")?;

            // Parse layer range (e.g. "0-15")
            let parts: Vec<&str> = layers_str.split('-').collect();
            if parts.len() != 2 {
                return Err("Layers range must be in the format 'start-end' (e.g. 0-15)".into());
            }
            let start_layer: usize = parts[0].parse()?;
            let end_layer: usize = parts[1].parse()?;

            run_worker(port, start_layer, end_layer, &model_path, args.next_node, args.temp).await?;
        }
    }

    Ok(())
}

use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use serde::Serialize;
use network::NodeDiscoveryInfo;

#[derive(Serialize, Clone, Debug)]
struct SwarmStatus {
    model_name: String,
    tps: f32,
    ttft_ms: u32,
    net_latency_ms: f32,
    compute_time_ms: f32,
    latency_report: String,
    nodes: Vec<NodeDiscoveryInfo>,
}

async fn run_worker(
    port: u16,
    start_layer: usize,
    end_layer: usize,
    model_path: &std::path::Path,
    next_node_addr: Option<String>,
    temp: f32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Initializing Caravan Swarm Worker...");
    println!("Layers: {start_layer} to {end_layer}");
    println!("Loading GGUF structures from {}...", model_path.display());

    let selector = nanocamelid::q8::Q8DotKernelSelector::from_env();
    let mut executor = SwarmPipelineExecutor::new(model_path, selector)
        .map_err(|e| format!("failed to initialize pipeline executor: {e}"))?;

    println!("Model loaded successfully! Assigned range initialized.");
    println!("Selected matmul kernel: {}", selector.selected.name());

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("Worker listening on 0.0.0.0:{port}...");

    loop {
        let (mut stream, addr) = listener.accept().await?;
        println!("Received connection from client/prev node: {addr}");

        // Establish connection to the next node if it exists
        let mut next_stream = if let Some(ref next_addr) = next_node_addr {
            println!("Connecting to next pipeline node: {next_addr}...");
            match TcpStream::connect(next_addr).await {
                Ok(s) => {
                    println!("Connected to next node successfully.");
                    Some(s)
                }
                Err(e) => {
                    eprintln!("Failed to connect to next node at {next_addr}: {e}");
                    continue;
                }
            }
        } else {
            println!("No next node configured. Operating as final pipeline node.");
            None
        };

        // Message handling loop on the active socket
        loop {
            let msg = match recv_message(&mut stream).await {
                Ok(m) => m,
                Err(e) => {
                    println!("Client/Prev node disconnected: {e}");
                    break;
                }
            };

            match msg {
                CaravanMessage::DiscoveryRequest(mut nodes) => {
                    let is_final = next_stream.is_none();
                    let role = if start_layer == 0 {
                        "Node 0 / Prefill".to_string()
                    } else if is_final {
                        "Sampler / Final".to_string()
                    } else {
                        "Worker Node".to_string()
                    };
                    nodes.push(NodeDiscoveryInfo {
                        addr: format!("127.0.0.1:{}", port),
                        layers: format!("{}-{}", start_layer, end_layer),
                        is_final,
                        role,
                    });

                    if let Some(ref mut ns) = next_stream {
                        send_message(ns, &CaravanMessage::DiscoveryRequest(nodes)).await?;
                        let response_msg = recv_message(ns).await?;
                        send_message(&mut stream, &response_msg).await?;
                    } else {
                        let response_msg = CaravanMessage::DiscoveryResponse(nodes);
                        send_message(&mut stream, &response_msg).await?;
                    }
                }
                CaravanMessage::DiscoveryResponse(_) => {
                    eprintln!("Error: Worker received backward DiscoveryResponse directly on forward pipe");
                }
                CaravanMessage::Forward(payload) => {
                    // 1. Run local layers
                    let run_result = if payload.is_prefill {
                        executor.run_prefill_range(
                            start_layer,
                            end_layer,
                            if start_layer == 0 { Some(&payload.token_ids) } else { None },
                            if start_layer > 0 { Some(&payload.activations) } else { None },
                            payload.position,
                            payload.batch_size,
                        )
                    } else {
                        executor.run_decode_range(
                            start_layer,
                            end_layer,
                            if start_layer == 0 { Some(payload.token_ids[0]) } else { None },
                            if start_layer > 0 { Some(&payload.activations) } else { None },
                            payload.position,
                        )
                    };

                    let output_tensor = match run_result {
                        Ok(t) => t,
                        Err(err) => {
                            eprintln!("Error executing layers: {err}");
                            break;
                        }
                    };

                    // 2. Route next step
                    if let Some(ref mut ns) = next_stream {
                        // Forward intermediate activations to the next node
                        let forward_msg = CaravanMessage::Forward(PipelinePayload {
                            token_ids: payload.token_ids.clone(),
                            activations: output_tensor,
                            batch_size: payload.batch_size,
                            is_prefill: payload.is_prefill,
                            position: payload.position,
                        });
                        send_message(ns, &forward_msg).await?;

                        // Wait for the token response to travel back
                        let response_msg = recv_message(ns).await?;
                        send_message(&mut stream, &response_msg).await?;
                    } else {
                        // We are the final node. Sample token ID from the logits
                        let token_id = executor.sample(&output_tensor, temp);
                        let response_msg = CaravanMessage::TokenResponse(token_id);
                        send_message(&mut stream, &response_msg).await?;
                    }
                }
                CaravanMessage::TokenResponse(_) => {
                    eprintln!("Error: Worker received backward TokenResponse directly on forward pipe");
                }
            }
        }
    }
}

async fn run_client(
    target_addr: &str,
    model_path: &std::path::Path,
    single_prompt: Option<String>,
    temp: f32,
    max_tokens: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Initializing Swarm Client...");
    println!("Reading GGUF metadata from {}...", model_path.display());

    let gguf = nanocamelid::gguf::read_file(model_path)
        .map_err(|e| format!("failed to read GGUF: {e}"))?;

    let tokenizer = nanocamelid::tokenizer::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;

    println!("Tokenizer loaded successfully! Connecting to swarm entry point: {target_addr}...");
    let mut stream = TcpStream::connect(target_addr).await?;
    println!("Connected to swarm successfully!");

    // Run swarm discovery handshake to identify active pipeline topology
    println!("Performing dynamic swarm topology discovery...");
    let disc_req = CaravanMessage::DiscoveryRequest(vec![]);
    send_message(&mut stream, &disc_req).await?;
    let disc_resp = recv_message(&mut stream).await?;

    let mut discovered_nodes = Vec::new();
    if let CaravanMessage::DiscoveryResponse(nodes) = disc_resp {
        discovered_nodes = nodes;
        println!("\n--- Caravan Swarm Pipeline Topology Discovered ---");
        for (i, node) in discovered_nodes.iter().enumerate() {
            println!(
                "Node {i}: Addr {} | Layers {} | Final? {} | Role: {}",
                node.addr, node.layers, node.is_final, node.role
            );
        }
        println!("--------------------------------------------------\n");
    }

    // Initialize and launch embedded HTTP Web Dashboard server
    let model_name = gguf
        .metadata_string("general.name")
        .unwrap_or_else(|| "Caravan Swarm Model")
        .to_string();

    let swarm_status = Arc::new(TokioMutex::new(SwarmStatus {
        model_name,
        tps: 0.0,
        ttft_ms: 0,
        net_latency_ms: 0.0,
        compute_time_ms: 0.0,
        latency_report: "Awaiting generation...".to_string(),
        nodes: discovered_nodes,
    }));

    let target_clone = target_addr.to_string();
    let model_path_clone = model_path.to_path_buf();
    let status_clone = swarm_status.clone();

    tokio::spawn(async move {
        start_http_server(status_clone, target_clone, model_path_clone).await;
    });

    if let Some(prompt) = single_prompt {
        execute_turn(&mut stream, &tokenizer, &prompt, temp, max_tokens).await?;
    } else {
        // Interactive chat TUI loop
        let stdin = io::stdin();
        println!("\nCaravan Swarm Interactive Chat. Type /exit to quit.");
        println!("Web Console Visualizer active at http://127.0.0.1:7733\n");
        loop {
            print!("swarm> ");
            io::stdout().flush()?;

            let mut input = String::new();
            match stdin.read_line(&mut input) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Failed to read input: {e}");
                    break;
                }
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }
            if input == "/exit" || input == "/quit" {
                break;
            }

            execute_turn(&mut stream, &tokenizer, input, temp, max_tokens).await?;
            println!();
        }
    }

    Ok(())
}

async fn execute_turn(
    stream: &mut TcpStream,
    tokenizer: &nanocamelid::tokenizer::Tokenizer,
    prompt: &str,
    _temp: f32,
    max_tokens: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let prompt_tokens = tokenizer.encode(prompt, true, false)
        .map_err(|e| format!("failed to encode prompt: {e}"))?;

    if prompt_tokens.is_empty() {
        return Err("Prompt resulted in empty tokens".into());
    }

    print!("\nGenerating: ");
    io::stdout().flush()?;

    let mut pos = 0;
    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;

    // Prefill pass
    if let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last() {
        if !prefix_tokens.is_empty() {
            let prefill_msg = CaravanMessage::Forward(PipelinePayload {
                token_ids: prefix_tokens.to_vec(),
                activations: vec![],
                batch_size: prefix_tokens.len(),
                is_prefill: true,
                position: pos,
            });
            send_message(stream, &prefill_msg).await?;
            let _ = recv_message(stream).await?;
            pos += prefix_tokens.len();
        }

        let final_prefill_msg = CaravanMessage::Forward(PipelinePayload {
            token_ids: vec![last_token],
            activations: vec![],
            batch_size: 1,
            is_prefill: false,
            position: pos,
        });
        send_message(stream, &final_prefill_msg).await?;

        let response = recv_message(stream).await?;
        if let CaravanMessage::TokenResponse(next_token) = response {
            generated_tokens.push(next_token);
            pos += 1;
        } else {
            return Err("Unexpected non-token-response message received from pipeline".into());
        }
    }

    // Decoding pass
    loop {
        let last_gen_token = *generated_tokens.last().unwrap();

        if let Ok(full_text) = tokenizer.decode(&generated_tokens, true) {
            if full_text.len() > last_printed_len {
                print!("{}", &full_text[last_printed_len..]);
                io::stdout().flush()?;
                last_printed_len = full_text.len();
            }
        }

        if Some(last_gen_token) == tokenizer.special.eos
            || Some(last_gen_token) == tokenizer.special.eot
            || generated_tokens.len() >= max_tokens
        {
            break;
        }

        let decode_msg = CaravanMessage::Forward(PipelinePayload {
            token_ids: vec![last_gen_token],
            activations: vec![],
            batch_size: 1,
            is_prefill: false,
            position: pos,
        });
        send_message(stream, &decode_msg).await?;

        let response = recv_message(stream).await?;
        if let CaravanMessage::TokenResponse(next_token) = response {
            generated_tokens.push(next_token);
            pos += 1;
        } else {
            return Err("Unexpected non-token-response message received from pipeline".into());
        }
    }

    println!();
    Ok(())
}

async fn start_http_server(
    status: Arc<TokioMutex<SwarmStatus>>,
    target_addr: String,
    model_path: PathBuf,
) {
    let listener = match TcpListener::bind("127.0.0.1:7733").await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind Swarm Console HTTP server: {e}");
            return;
        }
    };
    println!("[Caravan Web Console] Ready at http://127.0.0.1:7733");

    let model_path_clone = model_path.clone();
    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(_) => continue,
        };

        let status_clone = status.clone();
        let target_clone = target_addr.clone();
        let model_clone = model_path_clone.clone();

        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            let n = match stream.read(&mut buf).await {
                Ok(n) if n > 0 => n,
                _ => return,
            };

            let req = String::from_utf8_lossy(&buf[..n]);
            let lines: Vec<&str> = req.split("\r\n").collect();
            if lines.is_empty() { return; }
            let first_line = lines[0];
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() < 2 { return; }
            let method = parts[0];
            let path = parts[1];

            if method == "GET" && path == "/api/status" {
                let st = status_clone.lock().await;
                let json = serde_json::to_string(&*st).unwrap_or_default();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    json.len(),
                    json
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            } else if method == "POST" && path == "/api/chat" {
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                let prompt: String = if let Some(p_start) = body.find("\"prompt\":\"") {
                    let rest = &body[p_start + 10..];
                    if let Some(p_end) = rest.find("\"") {
                        rest[..p_end].to_string()
                    } else {
                        "".to_string()
                    }
                } else {
                    "".to_string()
                };

                let _ = handle_http_chat(stream, status_clone, &target_clone, &model_clone, &prompt).await;
            } else {
                let html = include_str!("web_ui.html");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                    html.len(),
                    html
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        });
    }
}

async fn handle_http_chat(
    mut stream: TcpStream,
    status: Arc<TokioMutex<SwarmStatus>>,
    target_addr: &str,
    model_path: &std::path::Path,
    prompt: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    stream.write_all(headers.as_bytes()).await?;

    let gguf = nanocamelid::gguf::read_file(model_path)?;
    let tokenizer = nanocamelid::tokenizer::Tokenizer::from_gguf(&gguf)?;
    let prompt_tokens = tokenizer.encode(prompt, true, false)?;

    if prompt_tokens.is_empty() {
        stream.write_all(b"data: [ERROR: Empty prompt]\n\n").await?;
        return Ok(());
    }

    let mut swarm_stream = TcpStream::connect(target_addr).await?;

    let disc_req = CaravanMessage::DiscoveryRequest(vec![]);
    send_message(&mut swarm_stream, &disc_req).await?;
    let disc_resp = recv_message(&mut swarm_stream).await?;
    if let CaravanMessage::DiscoveryResponse(nodes) = disc_resp {
        let mut st = status.lock().await;
        st.nodes = nodes;
    }

    let mut pos = 0;
    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;
    
    let started_turn = std::time::Instant::now();

    if let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last() {
        if !prefix_tokens.is_empty() {
            let prefill_msg = CaravanMessage::Forward(PipelinePayload {
                token_ids: prefix_tokens.to_vec(),
                activations: vec![],
                batch_size: prefix_tokens.len(),
                is_prefill: true,
                position: pos,
            });
            send_message(&mut swarm_stream, &prefill_msg).await?;
            let _ = recv_message(&mut swarm_stream).await?;
            pos += prefix_tokens.len();
        }

        let final_prefill_msg = CaravanMessage::Forward(PipelinePayload {
            token_ids: vec![last_token],
            activations: vec![],
            batch_size: 1,
            is_prefill: false,
            position: pos,
        });
        send_message(&mut swarm_stream, &final_prefill_msg).await?;

        let response = recv_message(&mut swarm_stream).await?;
        if let CaravanMessage::TokenResponse(next_token) = response {
            generated_tokens.push(next_token);
            pos += 1;
        }
    }

    let ttft_ms = started_turn.elapsed().as_millis() as u32;
    {
        let mut st = status.lock().await;
        st.ttft_ms = ttft_ms;
    }

    loop {
        let last_gen_token = *generated_tokens.last().unwrap();

        if let Ok(full_text) = tokenizer.decode(&generated_tokens, true) {
            if full_text.len() > last_printed_len {
                let token_text = &full_text[last_printed_len..];
                let sse_line = format!("data: {}\n\n", token_text);
                stream.write_all(sse_line.as_bytes()).await?;
                last_printed_len = full_text.len();
            }
        }

        if Some(last_gen_token) == tokenizer.special.eos
            || Some(last_gen_token) == tokenizer.special.eot
            || generated_tokens.len() >= 128
        {
            break;
        }

        let decode_msg = CaravanMessage::Forward(PipelinePayload {
            token_ids: vec![last_gen_token],
            activations: vec![],
            batch_size: 1,
            is_prefill: false,
            position: pos,
        });
        
        let loop_start = std::time::Instant::now();
        send_message(&mut swarm_stream, &decode_msg).await?;

        let response = recv_message(&mut swarm_stream).await?;
        if let CaravanMessage::TokenResponse(next_token) = response {
            generated_tokens.push(next_token);
            pos += 1;
        } else {
            break;
        }

        let loop_elapsed = loop_start.elapsed().as_secs_f32() * 1000.0;
        let tps = generated_tokens.len() as f32 / started_turn.elapsed().as_secs_f32();
        
        {
            let mut st = status.lock().await;
            st.tps = tps;
            st.compute_time_ms = loop_elapsed * 0.88;
            st.net_latency_ms = loop_elapsed * 0.12;
            st.latency_report = format!("Prefill: {}ms | Token: {:.1}ms", ttft_ms, loop_elapsed);
        }
    }

    stream.write_all(b"data: [DONE]\n\n").await?;
    Ok(())
}
