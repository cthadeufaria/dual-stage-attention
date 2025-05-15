use tch::{CModule, Tensor};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loading and Running a PyTorch Model in Rust: https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md
    let model: CModule = CModule::load("../../runs/models/DUAL_ATTENTION_LIVENFLX_II_2025-05-02_12:43:17_EPOCH_13")?;

    let input: Tensor = Tensor::randn(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));

    let output: Tensor = model.forward_ts(&[input])?;
    
    println!("Inference output: {:?}", output);
    
    Ok(())
}