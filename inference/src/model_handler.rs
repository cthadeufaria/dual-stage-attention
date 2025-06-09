// model_handler.rs
use tch::{CModule, Device, IValue, Tensor, nn::VarStore,};
use std::ffi::CString;
use libc::dlopen;

pub struct ModelHandler {
    model: CModule,
    device: Device,
}

impl ModelHandler {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let path = CString::new("/home/dev/repos/dual-stage-attention/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so").unwrap();
        unsafe {
            dlopen(path.into_raw(), 1);
        }

        let device = VarStore::new(Device::cuda_if_available()).device();
        println!("CUDA available: {}", tch::Cuda::is_available());
        println!("CUDA device count: {}", tch::Cuda::device_count());
        println!("Using device: {:?}", device);

        let model: CModule = CModule::load_on_device(
            model_path, 
            device
        )?;

        Ok(Self { model, device })
    }

    pub fn forward(&self, tensor1: Tensor, tensor2: Tensor, tensor3: Tensor, tensor4: Tensor) -> anyhow::Result<IValue> {
        let t1 = tensor1.to(self.device).to_kind(tch::Kind::Float);
        let t2 = tensor2.to(self.device).to_kind(tch::Kind::Float);
        let t3 = tensor3.to(self.device).to_kind(tch::Kind::Float);
        let t4 = tensor4.to(self.device).to_kind(tch::Kind::Float);

        let input = IValue::Tuple(vec![
            IValue::Tuple(vec![
                IValue::Tuple(vec![IValue::Tensor(t1), IValue::Tensor(t2)]),
                IValue::Tensor(t3),
            ]),
            IValue::Tensor(t4),
        ]);

        let output = self.model.forward_is(&[input])?;
        Ok(output)
    }
}
