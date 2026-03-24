//! ComputePipeline 캐시 — 셰이더 로드 + 파이프라인 생성

use std::collections::HashMap;
use wgpu::*;

use crate::gpu::GpuContext;

/// 셰이더 이름 → ComputePipeline 매핑
pub struct PipelineCache {
    pipelines: HashMap<String, ComputePipeline>,
    bind_group_layouts: HashMap<String, BindGroupLayout>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        }
    }

    /// WGSL 셰이더 소스에서 파이프라인 생성 + 캐시
    pub fn get_or_create(
        &mut self,
        gpu: &GpuContext,
        name: &str,
        wgsl_source: &str,
        entry_point: &str,
        bind_group_layout_entries: &[Vec<BindGroupLayoutEntry>],
    ) -> (&ComputePipeline, &BindGroupLayout) {
        if !self.pipelines.contains_key(name) {
            let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(name),
                source: ShaderSource::Wgsl(wgsl_source.into()),
            });

            // 현재는 단일 bind group 사용
            let bgl = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("{}_bgl", name)),
                entries: &bind_group_layout_entries[0],
            });

            let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", name)),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

            self.bind_group_layouts.insert(name.to_string(), bgl);
            self.pipelines.insert(name.to_string(), pipeline);
        }

        (
            self.pipelines.get(name).unwrap(),
            self.bind_group_layouts.get(name).unwrap(),
        )
    }

    pub fn get(&self, name: &str) -> Option<(&ComputePipeline, &BindGroupLayout)> {
        match (self.pipelines.get(name), self.bind_group_layouts.get(name)) {
            (Some(p), Some(b)) => Some((p, b)),
            _ => None,
        }
    }
}
