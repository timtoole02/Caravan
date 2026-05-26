use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PipelinePayload {
    /// Token IDs corresponding to the current forward pass, used for positional indexing in RoPE
    pub token_ids: Vec<u32>,
    /// Flattened activation matrix of dimensions [batch_size, hidden_dim]
    pub activations: Vec<f32>,
    /// Size of the batch in this step
    pub batch_size: usize,
    /// Whether this step is prompt prefilling or decoding
    pub is_prefill: bool,
    /// Absolute context position index in the KV cache
    pub position: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CaravanMessage {
    /// Sent forward through the pipeline nodes
    Forward(PipelinePayload),
    /// Returned from the final node back to the client
    TokenResponse(u32),
}

/// Send a length-prefixed serialized CaravanMessage over a TCP stream
pub async fn send_message(stream: &mut TcpStream, msg: &CaravanMessage) -> Result<(), Box<dyn Error + Send + Sync>> {
    let bytes = bincode::serialize(msg)?;
    let len = bytes.len() as u64;
    stream.write_u64(len).await?;
    stream.write_all(&bytes).await?;
    stream.flush().await?;
    Ok(())
}

/// Receive a length-prefixed serialized CaravanMessage from a TCP stream
pub async fn recv_message(stream: &mut TcpStream) -> Result<CaravanMessage, Box<dyn Error + Send + Sync>> {
    let len = stream.read_u64().await?;
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    let msg = bincode::deserialize(&buf)?;
    Ok(msg)
}
