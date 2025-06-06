use tch::{
    CModule, 
    Device, 
    Tensor, 
    IValue, 
    nn::VarStore,
};
use std::ffi::CString;
use libc::dlopen;
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

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Load a TorchScript model and perform inference using the `tch` crate in Rust.
//     // Solution to successfully using Cuda @ https://www.reddit.com/r/rust/comments/1j8vbww/torch_tchrs_with_cuda/.

//     let path = CString::new("/home/dev/repos/dual-stage-attention/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so").unwrap();
//     unsafe {
//         dlopen(path.into_raw(), 1);
//     }

//     let video_path = "/home/dev/repos/dual-stage-attention/datasets/LIVE_NFLX_Plus/assets_mp4_individual/AirShow_HuangBufferBasedAdaptor_Trace_0.mp4";

//     let vs = VarStore::new(Device::cuda_if_available());
//     println!("CUDA available: {}", tch::Cuda::is_available());
//     println!("CUDA device count: {}", tch::Cuda::device_count());
//     println!("Using device: {:?}", vs.device());

//     let model: CModule = CModule::load_on_device(
//         "/home/dev/repos/dual-stage-attention/runs/models/torchscript/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23.pt", 
//         vs.device()
//     )?;

//     // Create each tensor with the required shapes
//     let tensor1 = Tensor::randn(&[3, 80, 224, 224], (tch::Kind::Float, vs.device()));
//     let tensor2 = Tensor::randn(&[3, 320, 224, 224], (tch::Kind::Float, vs.device()));
//     let tensor3 = Tensor::randn(&[3, 10, 1080, 1920], (tch::Kind::Float, vs.device()));
//     let tensor4 = Tensor::randn(&[10, 4], (tch::Kind::Float, vs.device()));

//     // Build the nested tuple structure
//     let input = IValue::Tuple(vec![
//         IValue::Tuple(vec![
//             IValue::Tuple(vec![
//                 IValue::Tensor(tensor1),
//                 IValue::Tensor(tensor2)
//             ]),
//             IValue::Tensor(tensor3)
//         ]),
//         IValue::Tensor(tensor4)
//     ]);

//     println!("Inference input: {:?}", input);

//     let output: IValue = model.forward_is(&[input])?;

//     println!("Inference output: {:?}", output);

//     Ok(())
// }