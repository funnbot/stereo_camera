use anyhow::Result;
use libcamera::camera::ActiveCamera;
use libcamera::camera::Camera as LibCamera;
use libcamera::camera::CameraConfiguration;
use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraList;
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer_allocator::FrameBuffer;
use libcamera::framebuffer_allocator::FrameBufferAllocator;
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::stream;
use libcamera::stream::StreamRole;
use std::error::Error;
use std::rc::Rc;

pub struct Camera<'a> {
    cfg: Option<CameraConfiguration>,
    cam: ActiveCamera<'a>,
    alloc: FrameBufferAllocator,
    buffers: Vec<MemoryMappedFrameBuffer<FrameBuffer>>,
}

impl<'a> Camera<'a> {
    const PIXEL_FORMAT: PixelFormat =
        PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);
    const RESOLUTION: Size = Size {
        width: 4608,
        height: 2592,
    };

    fn generate_configuration(cam: &ActiveCamera) -> CameraConfiguration {
        let mut cfg = cam
            .generate_configuration(&[StreamRole::StillCapture])
            .expect("Still Capture is supported");

        let mut stream_cfg = cfg.get_mut(0).unwrap();
        stream_cfg.set_pixel_format(Self::PIXEL_FORMAT);
        stream_cfg.set_buffer_count(1);
        stream_cfg.set_size(Self::RESOLUTION);

        match cfg.validate() {
            CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
            CameraConfigurationStatus::Adjusted => {
                println!("Camera configuration was adjusted: {:?}", cfg)
            }
            CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
        }

        cfg
    }

    pub fn new(cam: ActiveCamera<'a>) -> Result<Self> {
        let alloc = FrameBufferAllocator::new(&cam);
        Ok(Self {
            cam,
            cfg: None,
            alloc,
            buffers: Vec::new(),
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.cfg = Some(Self::generate_configuration(&self.cam));
        let cfg = self.cfg.as_mut().unwrap();

        self.cam.configure(cfg)?;
        let stream_cfg = cfg.get(0).unwrap().stream().unwrap();
        self.buffers = self
            .alloc
            .alloc(&stream_cfg)?
            .into_iter()
            .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
            .collect();
        Ok(())
    }
}

pub struct StereoCamera<'a>(Camera<'a>, Camera<'a>);
impl<'a> StereoCamera<'a> {
    pub fn new(cam1: ActiveCamera<'a>, cam2: ActiveCamera<'a>) -> Result<Self> {
        Ok(Self(Camera::new(cam1)?, Camera::new(cam2)?))
    }
}

pub(crate) struct LibCameraSystem<'a> {
    manager: CameraManager,
    camera_list: Option<CameraList<'a>>,
    static_cams: Option<(LibCamera<'a>, LibCamera<'a>)>,
    cams: Option<(ActiveCamera<'a>, ActiveCamera<'a>)>,
    allocs: Option<(FrameBufferAllocator, FrameBufferAllocator)>,
}

impl<'a> LibCameraSystem<'a> {
    const PIXEL_FORMAT: PixelFormat =
        PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);
    const RESOLUTION: Size = Size {
        width: 4608,
        height: 2592,
    };

    pub fn new() -> Result<Self, Box<dyn Error>> {
        let manager = CameraManager::new()?;
        Ok(Self {
            manager,
            static_cams: None,
            allocs: None,
            cams: None,
            camera_list: None,
        })
    }

    fn generate_configuration(cam: &ActiveCamera) -> CameraConfiguration {
        let mut cfg = cam
            .generate_configuration(&[StreamRole::StillCapture])
            .expect("Still Capture is supported");

        let mut stream_cfg = cfg.get_mut(0).unwrap();
        stream_cfg.set_pixel_format(Self::PIXEL_FORMAT);
        stream_cfg.set_buffer_count(1);
        stream_cfg.set_size(Self::RESOLUTION);

        match cfg.validate() {
            CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
            CameraConfigurationStatus::Adjusted => {
                println!("Camera configuration was adjusted: {:?}", cfg)
            }
            CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
        }

        cfg
    }

    // fn alloc_buffers(cam: &ActiveCamera, alloc: &mut FrameBufferAllocator) -> Result<(), Box<dyn Error>> {
    //     let cfg = cam.camera();
    //     let stream_cfg = cfg.get(0).unwrap().stream().unwrap();
    //     alloc.alloc(&stream_cfg)?;
    //     Ok(())
    // }

    pub fn setup_cameras(&'a mut self) -> Result<(), Box<dyn Error>> {
        self.camera_list = Some(self.manager.cameras());

        self.static_cams = Some((
            self.camera_list.as_ref().unwrap().get(0).unwrap(),
            self.camera_list.as_ref().unwrap().get(1).unwrap(),
        ));

        self.cams = Some((
            self.static_cams.as_ref().unwrap().0.acquire()?,
            self.static_cams.as_ref().unwrap().1.acquire()?,
        ));

        let mut cfgs = (
            Self::generate_configuration(&self.cams.as_ref().unwrap().0),
            Self::generate_configuration(&self.cams.as_ref().unwrap().1),
        );

        let cams = self.cams.as_mut().unwrap();

        cams.0.configure(&mut cfgs.0)?;
        cams.1.configure(&mut cfgs.1)?;
        self.allocs = Some((
            FrameBufferAllocator::new(&cams.0),
            FrameBufferAllocator::new(&cams.1),
        ));
        let stream_cfgs = (
            cfgs.0.get(0).unwrap().stream().unwrap(),
            cfgs.1.get(0).unwrap().stream().unwrap(),
        );
        let buffers = (
            self.allocs.as_mut().unwrap().0.alloc(&stream_cfgs.0)?,
            self.allocs.as_mut().unwrap().1.alloc(&stream_cfgs.1)?,
        );
        let buffers = (
            buffers
                .0
                .into_iter()
                .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
                .collect::<Vec<_>>(),
            buffers
                .1
                .into_iter()
                .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
                .collect::<Vec<_>>(),
        );

        Ok(())
    }

    // pub fn capture(&mut self) -> Result<(), Box<dyn Error>> {
    //     let cams = self.cams.as_mut().unwrap();
    //     let allocs = self.allocs.as_mut().unwrap();
    //     let buffers = (
    //         allocs
    //             .0
    //             .alloc(&cams.0.camera().get(0).unwrap().stream().unwrap())?,
    //         allocs
    //             .1
    //             .alloc(&cams.1.camera().get(0).unwrap().stream().unwrap())?,
    //     );
    //     let buffers = (
    //         buffers
    //             .0
    //             .into_iter()
    //             .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
    //             .collect::<Vec<_>>(),
    //         buffers
    //             .1
    //             .into_iter()
    //             .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
    //             .collect::<Vec<_>>(),
    //     );

    //     Ok(())
    // }
}
