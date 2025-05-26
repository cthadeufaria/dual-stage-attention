use tch::{CModule, Tensor};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a TorchScript model and perform inference using the `tch` crate in Rust.
    // More information available @:
    // https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md
    // https://blog.gopenai.com/migrating-trained-pytorch-model-to-rust-a51869e8a51c
    // https://medium.com/@heyamit10/loading-and-running-a-pytorch-model-in-rust-f10d2577d570

    let model: CModule = CModule::load("/home/dev/repos/dual-stage-attention/runs/models/torchscript/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23.pt")?;

    let input: Tensor = Tensor::randn(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));

    let output: Tensor = model.forward_ts(&[input])?;

    println!("Inference output: {:?}", output);

    Ok(())
}