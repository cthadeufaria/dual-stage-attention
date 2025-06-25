use anyhow::{anyhow, Result};
use gstreamer::prelude::*;
use gstreamer::{ElementFactory, Pipeline, State};
use gstreamer_app::AppSink;
use tch::{
    Kind, Tensor
};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::time::Instant;

use crate::model_handler::ModelHandler;


pub fn start(model: Arc<ModelHandler>) -> Result<()> {
    let pipeline: Pipeline = create_server_pipeline(model)?;
    pipeline.set_state(State::Playing)?;

    // Bus message handling
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        use gstreamer::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                eprintln!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            _ => (),
        }
    }
    Ok(())
}

fn create_server_pipeline(model: Arc<ModelHandler>) -> Result<Pipeline> {
    gstreamer::init()?;
    let pipeline = Pipeline::new(Some("udp_server_pipeline"));

    // Create elements
    let udpsrc = ElementFactory::make("udpsrc")
        .property("port", 5000)
        .property("caps", gstreamer::Caps::builder("image/jpeg").build())
        .build()?;

    let jpegparse = ElementFactory::make("jpegparse").build()?;
    let jpegdec = ElementFactory::make("jpegdec").build()?;
    let videoconvert = ElementFactory::make("videoconvert").build()?;
    
    // Add capsfilter for RGB format
    let capsfilter = ElementFactory::make("capsfilter")
        .property("caps", gstreamer::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build())
        .build()?;

    let queue = ElementFactory::make("queue")
        .property("max-size-buffers", 1000u32)  // number of frames
        .property("max-size-bytes", 0u32)       // 0 = unlimited
        .property("max-size-time", 0u64)        // 0 = unlimited
        .build()?;

    // Create and configure appsink
    let appsink = ElementFactory::make("appsink")
        .property("emit-signals", true) // Enable signals
        .property("sync", false)       // No sync to playback
        .build()?;

    // Add elements to pipeline
    pipeline.add_many(&[
        &udpsrc, 
        &jpegparse, 
        &jpegdec, 
        &videoconvert,
        &capsfilter,
        &queue,
        &appsink
    ])?;

    // Link elements in order
    gstreamer::Element::link_many(&[
        &udpsrc,
        &jpegparse,
        &jpegdec,
        &queue,
        &videoconvert,
        &capsfilter,
        &appsink
    ])?;

    // Downcast to AppSink and configure callbacks
    let appsink = appsink.downcast::<AppSink>().map_err(|_| 
        anyhow!("Failed to downcast to AppSink")
    )?;

    let frame_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(320)));
    let buffer_clone = Arc::clone(&frame_buffer);
    let fast_stacked = Arc::new(Mutex::new(None::<Tensor>));  // Initially None
    let fast_stacked_clone = Arc::clone(&fast_stacked);

    let mean = Tensor::from_slice(&[0.485, 0.456, 0.406]).view([3, 1, 1]);
    let std = Tensor::from_slice(&[0.229, 0.224, 0.225]).view([3, 1, 1]);

    let mut frame_count = 0;
    let model_clone: Arc<ModelHandler> = Arc::clone(&model);
    let model_device = model_clone.device();

    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            // Called when a new sample is ready
            .new_sample(move |appsink| {
                // let start = Instant::now();

                let sample = appsink.pull_sample().map_err(|_| gstreamer::FlowError::Error)?;
                let buffer = sample.buffer().ok_or(gstreamer::FlowError::Error)?;
                let map = buffer.map_readable().map_err(|_| gstreamer::FlowError::Error)?;
                let data = map.as_slice();

                let caps = sample.caps().ok_or(gstreamer::FlowError::Error)?;
                let s = caps.structure(0).ok_or(gstreamer::FlowError::Error)?;

                let height = s.get::<i32>("height")
                    .map_err(|_| gstreamer::FlowError::Error)?;
                let width = s.get::<i32>("width")
                    .map_err(|_| gstreamer::FlowError::Error)?;

                // println!("Ingestion and preprocessing took: {:.2?}", start.elapsed());

                let tensor = Tensor::from_slice(data)
                    .to_kind(Kind::Uint8)
                    .reshape(&[height as i64, width as i64, 3])
                    .permute(&[2, 0, 1]); // (C, H, W)

                let normalized = (tensor.to_kind(Kind::Float) / 255.0 - &mean) / &std;

                static mut LAST_STALL_TIME: Option<Instant> = None;
                static mut LAST_BITRATE: f32 = 3000.0; // kbps
                static mut LAST_EVENT_TIME: Option<Instant> = None;

                let mut qos_features: Vec<[f32; 4]> = vec![];

                let mut buf = buffer_clone.lock().unwrap();
                if buf.len() == 320 {
                    buf.pop_front();
                }
                buf.push_back(normalized.shallow_clone());
                // println!("Buffer size: {}", buf.len());

                // println!("Received frame number: {}", frame_count);
                frame_count += 1;  // TODO: Use tokio to handle async frame processing.
                if buf.len() >= 320 && frame_count % 32 == 0 {  // TODO: Adapt the logic to update the tensors with only one second of data each iteration.
                    let start = Instant::now();

                    let slow_frames: Vec<_> = buf.iter()
                        .rev()               // Start from newest
                        .step_by(4)          // Pick every 4th going backwards
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()               // Restore chronological order
                        .map(|t| t.shallow_clone().to_device(model_device))
                        .collect();
                    let slow_stacked = Tensor::stack(&slow_frames, 1); // [3, 80, 224, 224]
                    let tensor1 = slow_stacked.upsample_bilinear2d(
                        &[224, 224], false, None, None
                    );

                    let mut fast_stacked_guard = fast_stacked_clone.lock().unwrap();
                    if let Some(existing) = &mut *fast_stacked_guard {
                        let new_frames: Vec<_> = buf
                            .iter()
                            .rev()
                            .take(32) // Take newest 32 frames
                            .map(|t| t.unsqueeze(1).to_device(model_device))
                            .collect();

                        let new_tensor = Tensor::cat(&new_frames, 1)
                            .upsample_bilinear2d(&[224, 224], false, None, None); // [3, 32, 224, 224]
                        let remaining_tensor = existing.narrow(1, 32, 288)
                            .upsample_bilinear2d(&[224, 224], false, None, None); // [3, 288, 224, 224]

                        let updated = Tensor::cat(&[&new_tensor, &remaining_tensor], 1); // [3, 320, 224, 224]

                        *existing = updated;

                    } else {
                        let fast_frames: Vec<_> = buf
                            .iter()
                            .map(|t| t.unsqueeze(1).upsample_bilinear2d(
                        &[224, 224], false, None, None
                    ).to_device(model_device))
                            .collect();
                        *fast_stacked_guard = Some(Tensor::cat(&fast_frames, 1));
                    }

                    let tensor2 = fast_stacked_guard.as_ref().unwrap().upsample_bilinear2d(
                        &[224, 224], false, None, None
                    );

                    let resnet_frames: Vec<_> = buf.iter()
                        .rev()
                        .step_by(32)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .map(|t| t.shallow_clone().to_device(model_device))
                        .collect();

                    let tensor3 = Tensor::stack(&resnet_frames, 1);

                    for _i in 0..10 {
                        // Simulated Playback Indicator: random stalling between 0 and 0.5s
                        let stall_duration = if rand::random::<f32>() < 0.2 {
                            // 20% chance of a stall
                            let d = rand::random::<f32>() * 0.5;
                            unsafe { LAST_STALL_TIME = Some(Instant::now()); }
                            d
                        } else {
                            0.0
                        };

                        // Simulated bitrate between 1000 and 5000 kbps
                        let new_bitrate: f32 = 1000.0 + rand::random::<f32>() * 4000.0;
                        let log_bitrate = new_bitrate.ln(); // RQ (log scale)

                        // Bitrate Switch (BS): only if drop from last
                        let bitrate_switch = unsafe {
                            let diff = LAST_BITRATE - new_bitrate;
                            let bs = if diff > 0.0 { diff.abs() } else { 0.0 };
                            LAST_BITRATE = new_bitrate;
                            bs
                        };

                        // TRF: Time since last drop or stall
                        let trf = unsafe {
                            let now = Instant::now();
                            let elapsed = if let Some(t) = LAST_EVENT_TIME {
                                now.duration_since(t).as_secs_f32()
                            } else {
                                0.0
                            };

                            if stall_duration > 0.0 || bitrate_switch > 0.0 {
                                LAST_EVENT_TIME = Some(now);
                            }

                            let normalized_trf = elapsed / 10.0; // Assume 10s video length for normalization
                            normalized_trf
                        };

                        qos_features.push([stall_duration, trf, log_bitrate, bitrate_switch]);
                    }

                    let tensor4 = Tensor::from_slice(&qos_features.concat())
                        .reshape(&[10, 4])
                        .to_kind(Kind::Float);

                    println!("Preprocessing + batching took: {:.2?}", start.elapsed());

                    let new_start = Instant::now();

                    let output = match model_clone.forward(tensor1, tensor2, tensor3, tensor4) {
                        Ok(out) => out,
                        Err(e) => {
                            eprintln!("Inference error: {}", e);
                            return Ok(gstreamer::FlowSuccess::Ok);  // or skip this frame gracefully
                        }
                    };

                    // println!("Inference output: {:?}", output);

                    println!("Inference took: {:.2?}", new_start.elapsed());
                    println!("Total processing time: {:.2?}", start.elapsed());
                }

                Ok(gstreamer::FlowSuccess::Ok)
            })
            .build()
    );

    Ok(pipeline)
}