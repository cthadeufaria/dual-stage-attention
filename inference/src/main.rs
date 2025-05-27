use tch::{CModule, Device, Tensor, IValue};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a TorchScript model and perform inference using the `tch` crate in Rust.
    let model: CModule = CModule::load_on_device(
        "/home/dev/repos/dual-stage-attention/runs/models/torchscript/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23.pt", 
        Device::Cpu
    )?;

    // Create each tensor with the required shapes
    let tensor1 = Tensor::randn(&[3, 216, 224, 224], (tch::Kind::Float, Device::Cpu));
    let tensor2 = Tensor::randn(&[3, 864, 224, 224], (tch::Kind::Float, Device::Cpu));
    let tensor3 = Tensor::randn(&[3, 27, 1080, 1920], (tch::Kind::Float, Device::Cpu));
    let tensor4 = Tensor::randn(&[27, 4], (tch::Kind::Float, Device::Cpu));

    // Build the nested tuple structure
    let input = IValue::Tuple(vec![
        IValue::Tuple(vec![
            IValue::Tuple(vec![
                IValue::Tensor(tensor1),
                IValue::Tensor(tensor2)
            ]),
            IValue::Tensor(tensor3)
        ]),
        IValue::Tensor(tensor4)
    ]);

    println!("Inference input: {:?}", input);

    let output: IValue = model.forward_is(&[input])?;

    println!("Inference output: {:?}", output);

    Ok(())
}