// ========================================
// Tensors over UDP:
// Python (Sender):

import numpy as np
import torch
import socket

def send_tensor(sock, tensor):
    arr = tensor.numpy().astype(np.float32)
    header = np.array([tensor.ndim] + list(tensor.shape), dtype=np.int32).tobytes()
    sock.sendall(header + arr.tobytes())

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_tensor(sock, tensor1)
// # repeat for other tensors...

// Rust (Receiver):
pub fn start() -> std::io::Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:5000")?;
    let mut buf = vec![0u8; 3 * 80 * 224 * 224 * 4]; // 4 bytes per float32
    println!("Server listening on port 5000...");

    let (amt, _src) = socket.recv_from(&mut buf)?;
    println!("Received {} bytes", amt);

    // Convert buffer to float32 tensor
    let float_data: Vec<f32> = buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let tensor = Tensor::from_slice(&float_data);

    println!("Tensor shape: {:?}", tensor.size());
    Ok(())
}

// Parse shape from first few bytes, reconstruct with Tensor::from_slice
// Example for one tensor:
let shape = vec![3, 80, 224, 224];
let tensor_data = &buf[header_len..];
let tensor = Tensor::from_slice(&f32_data).reshape(&shape[..]);

// =========================================
// Tensors locally saved and loaded (can load from cached dataset / need to load from Bisect video):
// Python (Sender):

import torch

tensor1 = torch.randn(3, 80, 224, 224)
tensor2 = torch.randn(3, 320, 224, 224)
tensor3 = torch.randn(3, 10, 1080, 1920)
tensor4 = torch.randn(10, 4)

bundle = ((tensor1, tensor2), tensor3), tensor4
torch.save(bundle, "input.pt")

Rust (Receiver):

let input_ivalue = tch::IValue::load("input.pt")?;


// ========================================================
// timestamped creation of tensor in Rust:
let buffer_pts = buffer.pts().ok_or(gstreamer::FlowError::Error)?; // PTS in nanoseconds

let mut last_time = last_slow_sample_time.lock().unwrap();
let time_diff = match *last_time {
    Some(last) => buffer_pts.nseconds() - last,
    None => slow_sample_interval_ns,
};

if time_diff >= slow_sample_interval_ns {
    *last_time = Some(buffer_pts.nseconds().unwrap());

    let mut slow_buf = slow_buffer_clone.lock().unwrap();
    if slow_buf.len() == 80 {
        slow_buf.pop_front();
    }

    slow_buf.push_back(normalized.shallow_clone());

    if slow_buf.len() == 80 {
        let stacked = Tensor::stack(&slow_buf.iter().collect::<Vec<_>>(), 1); // [3, 80, 224, 224]
        println!("Slow pathway input shape: {:?}", stacked.size());
    }
}

// ===========================================
    let slow_pathway_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(320)));
    let slow_buffer_clone = Arc::clone(&slow_pathway_buffer);

    let frame_counter = Arc::new(Mutex::new(0usize));
    let counter_clone = Arc::clone(&frame_counter);

                let mut count = counter_clone.lock().unwrap();
                let should_sample = *count % 30 == 0;
                let should_sample_slow = *count % 4 == 0;

// <!-- create resnet50 and slow pathway input (timestamp misaligned)
// ===========================
    *count += 1;

    if should_sample {  // TODO: If the video finishes. clear buffer.
        let mut buf = buffer_clone.lock().unwrap();
        if buf.len() == 10 {
            buf.pop_front(); // Remove the oldest frame
        }

        buf.push_back(normalized.shallow_clone());

        if buf.len() == 10 {
            let stacked = Tensor::stack(&buf.iter().collect::<Vec<_>>(), 1);
            println!("ResNet50 input shape: {:?}", stacked.size());
            // You can now insert this into your IValue structure
            // Here you would call your model inference function
            // For example:
            // let output = model.forward_is(&[IValue::Tensor(stacked)])?;
            // println!("3. Inference output: {:?}", stacked);
        }
    }

    if should_sample_slow {
        let mut slow_buf = slow_buffer_clone.lock().unwrap();
        if slow_buf.len() == 80 {
            slow_buf.pop_front();
        }

        slow_buf.push_back(normalized.shallow_clone());

        if slow_buf.len() == 80 {
            let stacked = Tensor::stack(&slow_buf.iter().collect::<Vec<_>>(), 1); // Shape: [3, 80, 224, 224]
            println!("Slow pathway input shape: {:?}", stacked.size());

            // TODO: Store into IValue::Tensor or pass to model here
        }



// <!-- use torchscript model in Rust
// ===========================
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a TorchScript model and perform inference using the `tch` crate in Rust.
    // Solution to successfully using Cuda @ https://www.reddit.com/r/rust/comments/1j8vbww/torch_tchrs_with_cuda/.

    let path = CString::new("/home/dev/repos/dual-stage-attention/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so").unwrap();
    unsafe {
        dlopen(path.into_raw(), 1);
    }

    let video_path = "/home/dev/repos/dual-stage-attention/datasets/LIVE_NFLX_Plus/assets_mp4_individual/AirShow_HuangBufferBasedAdaptor_Trace_0.mp4";

    let vs = VarStore::new(Device::cuda_if_available());
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    println!("Using device: {:?}", vs.device());

    let model: CModule = CModule::load_on_device(
        "/home/dev/repos/dual-stage-attention/runs/models/torchscript/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23.pt", 
        vs.device()
    )?;

    // Create each tensor with the required shapes
    let tensor1 = Tensor::randn(&[3, 80, 224, 224], (tch::Kind::Float, vs.device()));
    let tensor2 = Tensor::randn(&[3, 320, 224, 224], (tch::Kind::Float, vs.device()));
    let tensor3 = Tensor::randn(&[3, 10, 1080, 1920], (tch::Kind::Float, vs.device()));
    let tensor4 = Tensor::randn(&[10, 4], (tch::Kind::Float, vs.device()));

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