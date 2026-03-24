//! wgpu Device/Queue/Adapter 초기화

use anyhow::{bail, Result};
use wgpu::*;

/// GPU 컨텍스트 — wgpu 장치 + 큐
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_name: String,
    pub backend: Backend,
}

impl GpuContext {
    /// 최적 GPU 어댑터를 선택하여 초기화
    pub fn new(preferred_backend: Option<Backend>) -> Result<Self> {
        pollster::block_on(Self::new_async(preferred_backend))
    }

    async fn new_async(preferred_backend: Option<Backend>) -> Result<Self> {
        let backends = match preferred_backend {
            Some(Backend::Vulkan) => Backends::VULKAN,
            Some(Backend::Metal) => Backends::METAL,
            Some(Backend::Dx12) => Backends::DX12,
            _ => Backends::all(),
        };

        let instance = Instance::new(&InstanceDescriptor {
            backends,
            flags: InstanceFlags::from_build_config()
                | InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });

        // 먼저 일반 어댑터 시도, 실패 시 비준수 어댑터도 허용
        // (WSL2 dozen = D3D12→Vulkan 번역 레이어, non-compliant로 분류됨)
        let adapter = match instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) if a.get_info().device_type != DeviceType::Cpu => Some(a),
            _ => {
                // CPU fallback 또는 어댑터 없음 → 모든 어댑터 중 GPU 직접 탐색
                eprintln!("표준 어댑터에서 GPU 미발견, 비준수 어댑터 탐색 중...");
                let all_adapters = instance.enumerate_adapters(Backends::all());
                let mut gpu_adapter = None;
                for a in all_adapters {
                    let info = a.get_info();
                    eprintln!("  발견: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
                    if info.device_type == DeviceType::DiscreteGpu
                        || info.device_type == DeviceType::IntegratedGpu
                    {
                        gpu_adapter = Some(a);
                        break;
                    }
                }
                gpu_adapter
            }
        };

        let adapter = match adapter {
            Some(a) => a,
            None => bail!("wgpu 어댑터를 찾을 수 없음"),
        };

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = info.backend;

        log::info!(
            "GPU 어댑터: {} ({:?}), driver: {}",
            adapter_name,
            backend,
            info.driver
        );

        // 요청 limits: 기본값 사용 (대부분의 GPU에서 충분)
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("dense-editor"),
                    required_features: Features::PUSH_CONSTANTS,
                    required_limits: Limits {
                        max_push_constant_size: 128,
                        ..Limits::default()
                    },
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("GPU 장치 요청 실패: {}", e))?;

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
        })
    }
}
