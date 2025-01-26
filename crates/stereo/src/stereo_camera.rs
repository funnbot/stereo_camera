use anyhow::Result;
use libcamera::camera::ActiveCamera;
use libcamera::camera::Camera as LibCamera;
use libcamera::camera::CameraConfiguration;
use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraList;
use libcamera::camera_manager::CameraManager;
use libcamera::control::Control;
use libcamera::control::ControlList;
use libcamera::controls as ctrls;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer_allocator::FrameBuffer;
use libcamera::framebuffer_allocator::FrameBufferAllocator;
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::Request;
use libcamera::request::ReuseFlag;
use libcamera::stream;
use libcamera::stream::Stream;
use libcamera::stream::StreamRole;
use std::cell::Cell;
use std::error::Error;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

type Buffer = MemoryMappedFrameBuffer<FrameBuffer>;

pub struct Camera<'a> {
    cfg: CameraConfiguration,
    cam: ActiveCamera<'a>,
    rx: Option<mpsc::Receiver<Request>>,
    stream: Option<Stream>,
}

impl<'a> Camera<'a> {
    const PIXEL_FORMAT: PixelFormat =
        PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);
    pub const FRAME_SIZE: Size = Size {
        width: 4608,
        height: 2592,
    };
    const BUFFER_COUNT: u32 = 1;

    fn generate_configuration(cam: &ActiveCamera) -> CameraConfiguration {
        let mut cfg = cam
            .generate_configuration(&[StreamRole::StillCapture])
            .expect("Still Capture is supported");

        let mut stream_cfg = cfg.get_mut(0).unwrap();
        stream_cfg.set_pixel_format(Self::PIXEL_FORMAT);
        stream_cfg.set_buffer_count(Self::BUFFER_COUNT);
        stream_cfg.set_size(Self::FRAME_SIZE);

        match cfg.validate() {
            CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
            CameraConfigurationStatus::Adjusted => {
                println!("Camera configuration was adjusted: {:?}", cfg)
            }
            CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
        }

        cfg
    }

    pub fn new(mut cam: ActiveCamera<'a>) -> Result<Self> {
        let mut cfg = Self::generate_configuration(&cam);
        cam.configure(&mut cfg)?;
        Ok(Self {
            cam,
            rx: None,
            cfg,
            stream: None,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        // setup request handling
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.cam.on_request_completed(move |req| {
            tx.send(req).unwrap();
        });

        let mut controls = ControlList::new();
        controls.set(ctrls::AeEnable(false))?;
        controls.set(ctrls::ExposureTime(10_000))?;
        // controls.set(ctrls::AwbEnable(false))?;
        self.cam.start(Some(&controls))?;
        Ok(())
    }

    pub fn stream(&self) -> &Stream {
        self.stream.as_ref().unwrap()
    }

    pub fn create_requests(&mut self) -> Result<Vec<Request>> {
        let stream = self.cfg.get(0).unwrap().stream().unwrap();
        let mut alloc = FrameBufferAllocator::new(&self.cam);
        let buffers = alloc
            .alloc(&stream)?
            .into_iter()
            .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
            .collect::<Vec<_>>();

        let reqs = buffers
            .into_iter()
            .map(|buf| {
                let mut req = self.cam.create_request(None).unwrap();
                req.add_buffer(&stream, buf).unwrap();
                req
            })
            .collect();

        self.stream = Some(stream);
        Ok(reqs)
    }

    pub fn queue_request(&mut self, mut request: Request) -> Result<()> {
        request.reuse(ReuseFlag::REUSE_BUFFERS);
        self.cam.queue_request(request)?;
        Ok(())
    }

    #[must_use]
    pub fn wait_capture(&mut self) -> Result<Request> {
        let rx = self.rx.as_ref().unwrap();
        let mut request = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        Ok(request)
    }

    pub fn extract_frame<'b>(&self, request: &'b Request) -> Result<&'b [u8]> {
        let buf: &Buffer = request.buffer(self.stream()).unwrap();
        let &rgb_data = buf.data().get(0).unwrap();
        Ok(rgb_data)
    }

    pub fn convert_frame_to_cv(&self, frame: &[u8]) -> Result<opencv::core::Mat> {
        let size = opencv::core::Size {
            width: Self::FRAME_SIZE.width as i32,
            height: Self::FRAME_SIZE.height as i32,
        };

        let converted_data: &[opencv::core::Vec3b] = unsafe {
            std::slice::from_raw_parts(
                frame.as_ptr() as *const opencv::core::Vec3b,
                size.width as usize * size.height as usize,
            )
        };

        let img = opencv::core::Mat::new_nd_with_data(&[size.height, size.width], converted_data)
            .unwrap();
        let mut dest = opencv::core::Mat::default();
        opencv::imgproc::cvt_color_def(&img, &mut dest, opencv::imgproc::COLOR_RGB2BGR).unwrap();
        Ok(dest)
    }
}

pub struct StereoCamera<'a>(Camera<'a>, Camera<'a>);
impl<'a> StereoCamera<'a> {
    pub fn new(cam1: ActiveCamera<'a>, cam2: ActiveCamera<'a>) -> Result<Self> {
        Ok(Self(Camera::new(cam1)?, Camera::new(cam2)?))
    }
}
