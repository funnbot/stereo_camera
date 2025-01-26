mod stereo_camera;

use anyhow::{Ok, Result};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    controls as ctrls,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    logging::LoggingLevel,
    pixel_format::{PixelFormat, PixelFormats},
    properties,
    stream::StreamRole,
};
use opencv::{
    self as cv,
    boxed_ref::BoxedRef,
    core::{Mat, MatTrait, MatTraitConst, MatTraitConstManual, Size as CVSize, Vec3b},
};
use std::{fmt::Debug, time::Duration};
use std::{fmt::Write as _, thread};
use stereo_camera::{Camera, StereoCamera};

// drm-fourcc does not have MJPEG type yet, construct it from raw fourcc identifier
const PIXEL_FORMAT_MJPEG: PixelFormat =
    PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);

fn main() {
    program().unwrap();
}

fn program() -> Result<()> {
    let manager = CameraManager::new()?;
    let cameras = manager.cameras();
    let cams = (
        cameras.get(0).expect("Camera 0 exists"),
        cameras.get(1).expect("Camera 1 exists"),
    );
    let cams = (cams.0.acquire()?, cams.1.acquire()?);
    let mut cams = (Camera::new(cams.0)?, Camera::new(cams.1)?);

    let mut reqs = (cams.0.create_requests()?, cams.1.create_requests()?);
    let reqs = (
        reqs.0.pop().expect("Request exists"),
        reqs.1.pop().expect("Request exists"),
    );

    cams.0.start()?;
    cams.1.start()?;

    thread::sleep(Duration::from_secs(2));

    cams.0.queue_request(reqs.0)?;
    cams.1.queue_request(reqs.1)?;

    let mut reqs = (cams.0.wait_capture()?, cams.1.wait_capture()?);

    let mut min = Duration::MAX;
    let mut max = Duration::default();
    let mut sum = Duration::default();

    for i in 0..10 {
        cams.0.queue_request(reqs.0)?;
        cams.1.queue_request(reqs.1)?;

        reqs = (cams.0.wait_capture()?, cams.1.wait_capture()?);
        if i > 4 {
            let stamps: (i64, i64) = (
                reqs.0.metadata().get::<ctrls::SensorTimestamp>()?.0,
                reqs.1.metadata().get::<ctrls::SensorTimestamp>()?.0,
            );
            let stamps = (
                Duration::from_nanos(stamps.0 as u64),
                Duration::from_nanos(stamps.1 as u64),
            );
            let diff = stamps.0.abs_diff(stamps.1);
            min = min.min(diff);
            max = max.max(diff);
            sum += diff;
        }
        //println!("Capture difference: {:?}", stamps.0.abs_diff(stamps.1));
        //thread::sleep(Duration::from_millis(100));
    }

    println!("Min: {:?}, Max: {:?}, Avg: {:?}", min, max, sum / 100);

    let frames = (
        cams.0.extract_frame(&reqs.0)?,
        cams.1.extract_frame(&reqs.1)?,
    );

    let imgs = (
        cams.0.convert_frame_to_cv(frames.0)?,
        cams.1.convert_frame_to_cv(frames.1)?,
    );

    cv::imgcodecs::imwrite_def("out1.png", &imgs.0).unwrap();
    cv::imgcodecs::imwrite_def("out2.png", &imgs.1).unwrap();

    Ok(())
}

// fn main2() {
//     let filename = std::env::args()
//         .nth(1)
//         .expect("Usage ./jpeg_capture <filename.jpg>");

//     let mgr = CameraManager::new().unwrap();
//     let cameras = mgr.cameras();
//     let cam = cameras.get(0).expect("No cameras found");

//     println!(
//         "Using camera: {}",
//         *cam.properties().get::<properties::Model>().unwrap()
//     );

//     let mut cam = cam.acquire().expect("Unable to acquire camera");

//     // This will generate default configuration for each specified role
//     let mut cfgs = cam
//         .generate_configuration(&[StreamRole::ViewFinder])
//         .unwrap();

//     // Use MJPEG format so we can write resulting frame directly into jpeg file
//     cfgs.get_mut(0)
//         .unwrap()
//         .set_pixel_format(PIXEL_FORMAT_MJPEG);
//     let size = Size {
//         width: 4608,
//         height: 2592,
//     };
//     cfgs.get_mut(0).unwrap().set_size(size);
//     println!("Generated config: {:#?}", cfgs);

//     match cfgs.validate() {
//         CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
//         CameraConfigurationStatus::Adjusted => {
//             println!("Camera configuration was adjusted: {:?}", cfgs)
//         }
//         CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
//     }

//     // Ensure that pixel format was unchanged
//     assert_eq!(
//         cfgs.get(0).unwrap().get_pixel_format(),
//         PIXEL_FORMAT_MJPEG,
//         "MJPEG is not supported by the camera"
//     );

//     cam.configure(&mut cfgs)
//         .expect("Unable to configure camera");

//     let mut alloc = FrameBufferAllocator::new(&cam);

//     // Allocate frame buffers for the stream
//     let cfg = cfgs.get(0).unwrap();
//     let stream = cfg.stream().unwrap();
//     let buffers = alloc.alloc(&stream).unwrap();
//     println!("Allocated {} buffers", buffers.len());

//     // Convert FrameBuffer to MemoryMappedFrameBuffer, which allows reading &[u8]
//     let buffers = buffers
//         .into_iter()
//         .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
//         .collect::<Vec<_>>();

//     // Create capture requests and attach buffers
//     let mut reqs = buffers
//         .into_iter()
//         .map(|buf| {
//             let mut req = cam.create_request(None).unwrap();
//             req.add_buffer(&stream, buf).unwrap();
//             req
//         })
//         .collect::<Vec<_>>();

//     // Completed capture requests are returned as a callback
//     let (tx, rx) = std::sync::mpsc::channel();
//     cam.on_request_completed(move |req| {
//         tx.send(req).unwrap();
//     });

//     cam.start(None).unwrap();

//     // Multiple requests can be queued at a time, but for this example we just want a single frame.
//     cam.queue_request(reqs.pop().unwrap()).unwrap();

//     println!("Waiting for camera request execution");
//     let req = rx
//         .recv_timeout(Duration::from_secs(2))
//         .expect("Camera request failed");

//     println!("Camera request {:?} completed!", req);

//     // Get framebuffer for our stream
//     let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
//     println!("FrameBuffer metadata: {:#?}", framebuffer.metadata());

//     // MJPEG format has only one data plane containing encoded jpeg data with all the headers
//     let planes = framebuffer.data();
//     let rgb_data = planes.get(0).unwrap();

//     // Actual JPEG-encoded data will be smalled than framebuffer size, its length can be obtained from metadata.
//     let jpeg_len = framebuffer
//         .metadata()
//         .unwrap()
//         .planes()
//         .get(0)
//         .unwrap()
//         .bytes_used as usize;

//     let size = CVSize {
//         width: 4608,
//         height: 2592,
//     };

//     let converted_data: &[Vec3b] = unsafe {
//         std::slice::from_raw_parts(
//             rgb_data.as_ptr() as *const Vec3b,
//             size.width as usize * size.height as usize,
//         )
//     };

//     let img = Mat::new_nd_with_data(&[size.height, size.width], converted_data).unwrap();
//     // let img = mat.reshape(3, size.height).unwrap();
//     // let mat = Mat::new_size_with_data::<u8>(
//     //     CVSize {
//     //         width: 4608,
//     //         height: 2592,
//     //     },
//     //     rgb_data,
//     // )
//     // .unwrap();
//     // println!("rgb image: {:#?}", DebugPrintImage(&mat));
//     let mut dest = cv::core::Mat::default();
//     cv::imgproc::cvt_color_def(&img, &mut dest, cv::imgproc::COLOR_RGB2BGR).unwrap();
//     // println!("rgb image: {:#?}", DebugPrintImage(&dest));
//     cv::imgcodecs::imwrite_def(&filename, &dest).unwrap();
//     // println!("Written {} bytes to {}", jpeg_len, &filename);

//     // Everything is cleaned up automatically by Drop implementations
// }

struct DebugPrintImage<'a, T>(&'a T)
where
    T: MatTraitConst + MatTraitConstManual;

impl<'a, T> Debug for DebugPrintImage<'a, T>
where
    T: MatTraitConst + MatTraitConstManual,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let img = self.0;
        let size = img.size().unwrap();
        let shape = img.mat_size();
        let dims = img.dims();
        let channels = img.channels();
        let depth = img.depth();
        let step = img.step1_def().unwrap();
        let elem_size = img.elem_size().unwrap();
        let elem_size1 = img.elem_size1();

        assert!(
            shape.len() == 2 || shape.len() == 3,
            "only 2D images of 1 or more channels supported"
        );
        let (rows, cols) = (shape[0], shape[1]);
        assert!(
            rows == size.height && cols == size.width,
            "shape and size should match"
        );
        let pixel_len = if shape.len() == 3 { shape[2] } else { 1 };
        assert!(pixel_len <= 4, "");
        // assert!(channels == pixel_len, "channels should match pixel len");

        // opencv stores matrices in row-major, so each row is contiguous in memory
        // however, indexing using Mat::at() is at(row, col) or at(y, x)
        // the size() of a Mat is (rows, cols) or (height, width)
        // so the step is the number of bytes to get to the next row

        // preview the data, much too big to print all of it
        // instead grab the first 5 and last 5 values of the first 4 rows
        // [[0, 1, 2, 3, ..., 253, 254, 255],
        // [0, 1, 2, 3, ..., 253, 254, 255],
        // ^ again for row 3
        // ^ again for row 4
        // ..., <-- skip all of the rows in between
        // [0, 1, 2, 3, ..., 253, 254, 255],
        // ^ again for row N-3
        // ^ again for row N-2
        // ^ again for row N-1
        // [0, 1, 2, 3, ..., 253, 254, 255]]
        //
        // if the type is not a scalar, could also be printer as tuple (a, b)

        let mut data_str = String::from("[");
        let write_pixel = |data_str: &mut String, row: i32, col: i32| {
            let get = |i2: i32| img.at_3d::<u8>(row, col, i2).unwrap();
            match pixel_len {
                1 => {}
                2 => {}
                3 => {
                    write!(data_str, "({}, {}, {})", get(0), get(1), get(2)).unwrap();
                }
                4 => {}
                _ => {}
            }
        };

        for row in 0..std::cmp::min(10, rows) {
            const COLS: [i32; 8] = [0, 1, 2, 3, -4, -3, -2, -1];
            for col in COLS {
                let idx = if col >= 0 { col } else { cols + col };
                write_pixel(&mut data_str, row, idx);
                if col != -1 {
                    write!(data_str, ", ")?;
                }
                if col == 3 {
                    write!(data_str, " ..., ")?;
                }
            }
            writeln!(data_str, "],")?;
        }

        f.debug_struct("Mat")
            .field("size", &size)
            .field("shape", &shape)
            .field("dims", &dims)
            .field("channels", &channels)
            //.field("pixel_len", &pixel_len)
            .field("depth", &depth)
            .field("step", &step)
            .field("elem_size", &elem_size)
            .field("elem_size1", &elem_size1)
            .field("data", &data_str)
            .finish()
    }
}
