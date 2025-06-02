use anyhow::{anyhow, Result};
use gstreamer::prelude::*;
use gstreamer::{ElementFactory, Pipeline, State};
use gstreamer_app::AppSink; // Required for appsink functionality
use gstreamer_app::prelude::*; // For AppSink methods

pub fn start() -> Result<()> {
    let pipeline: Pipeline = create_server_pipeline()?;
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

fn create_server_pipeline() -> Result<Pipeline> {
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
        &appsink
    ])?;

    // Link elements in order
    gstreamer::Element::link_many(&[
        &udpsrc,
        &jpegparse,
        &jpegdec,
        &videoconvert,
        &capsfilter,
        &appsink
    ])?;

    // Downcast to AppSink and configure callbacks
    let appsink = appsink.downcast::<AppSink>().map_err(|_| 
        anyhow!("Failed to downcast to AppSink")
    )?;

    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            // Called when a new sample is ready
            .new_sample(move |appsink| {
                // Try to pull the sample
                let sample = appsink.pull_sample().map_err(|_| {
                    eprintln!("Failed to pull sample");
                    gstreamer::FlowError::Error
                })?;

                // Get the buffer from the sample
                let buffer = sample.buffer().ok_or_else(|| {
                    eprintln!("Sample has no buffer");
                    gstreamer::FlowError::Error
                })?;

                // Print basic buffer info
                println!("Received buffer of size: {} bytes", buffer.size());

                // Access buffer data
                let map = buffer.map_readable().map_err(|_| {
                    eprintln!("Failed to map buffer");
                    gstreamer::FlowError::Error
                })?;

                // Print first 3 bytes (RGB values)
                let data = map.as_slice();
                if data.len() >= 3 {
                    println!("First pixel RGB: {:?}", &data[0..3]);
                }

                Ok(gstreamer::FlowSuccess::Ok)
            })
            .build()
    );

    Ok(pipeline)
}