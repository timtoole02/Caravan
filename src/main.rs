use clap::{Parser, Subcommand};
use std::error::Error;
use std::path::PathBuf;
use tokio::net::{TcpListener, TcpStream};
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
                        let token_id = if payload.is_prefill {
                            executor.sample(&output_tensor, temp)
                        } else {
                            executor.sample(&output_tensor, temp)
                        };

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

    if let Some(prompt) = single_prompt {
        execute_turn(&mut stream, &tokenizer, &prompt, temp, max_tokens).await?;
    } else {
        // Interactive chat TUI loop
        let stdin = io::stdin();
        println!("\nCaravan Swarm Interactive Chat. Type /exit to quit.\n");
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
    // Render simple chat formatting or just encode prompt
    // Let's perform standard tokenizer encoding
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
        // Send prefill chunks if prefix exists
        if !prefix_tokens.is_empty() {
            // Send entire prefix as a batch prefill
            let prefill_msg = CaravanMessage::Forward(PipelinePayload {
                token_ids: prefix_tokens.to_vec(),
                activations: vec![],
                batch_size: prefix_tokens.len(),
                is_prefill: true,
                position: pos,
            });
            send_message(stream, &prefill_msg).await?;
            // Await dummy/acknowledged token response to maintain sync
            let _ = recv_message(stream).await?;
            pos += prefix_tokens.len();
        }

        // Send the last token to trigger the first actual sampled token
        let final_prefill_msg = CaravanMessage::Forward(PipelinePayload {
            token_ids: vec![last_token],
            activations: vec![],
            batch_size: 1,
            is_prefill: false,
            position: pos,
        });
        send_message(stream, &final_prefill_msg).await?;

        // Receives the first generated token ID from the swarm final node
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

        // Print incremental decoded text
        if let Ok(full_text) = tokenizer.decode(&generated_tokens, true) {
            if full_text.len() > last_printed_len {
                print!("{}", &full_text[last_printed_len..]);
                io::stdout().flush()?;
                last_printed_len = full_text.len();
            }
        }

        // Check completion conditions
        if Some(last_gen_token) == tokenizer.special.eos
            || Some(last_gen_token) == tokenizer.special.eot
            || generated_tokens.len() >= max_tokens
        {
            break;
        }

        // Forward decode pass for the next token
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
