mod udp_server;
mod model_handler;
use model_handler::ModelHandler;
use std::sync::Arc;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/dev/repos/dual-stage-attention/runs/models/torchscript/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23.pt";

    let model_handler = Arc::new(ModelHandler::new(model_path)?);
    udp_server::start(model_handler)?;
    Ok(())
}