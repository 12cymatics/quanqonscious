# Capability matrix — required evidence, not a hardware inventory

Status: all project-device capability cells are UNVERIFIED. Do not infer the current Mac from an older machine mentioned in conversation. Research date 2026-10-23.

| Backend | Inspect before choosing mapping | Correctness/precision constraints | Evidence to retain |
|---|---|---|---|
| CPU reference | Architecture, cores, cache hierarchy, available SIMD, compiler vectorization report, memory bandwidth | Exact-domain library support; overflow bounds; alignment; reduction and fast-math settings | CPU/OS/compiler version, build flags, exact tests, access-pattern timings |
| WebGPU | Adapter and device availability; features; compute group/count/storage/buffer limits; subgroup support only if advertised; timestamp support | WGSL type/operation support, no assumption of native arbitrary precision; workgroup-scope barriers; uniform participation | Feature/limit JSON, shader validation, real-device result checks, browser/backend version, device-loss tests |
| Metal | Device GPU family and OS/API availability; pipeline threadExecutionWidth; pipeline maximum threads; memory requirements; nonuniform-group feature support | Arithmetic and atomics support per feature family; explicit memory synchronization; exact arithmetic only through validated representation | Family/features, pipeline properties, compiler options, device result checks, launch shapes |
| CUDA (optional non-Mac evidence path) | Actual NVIDIA device and compute capability, toolkit/driver, block/grid and shared-memory limits, register pressure | CUDA-specific warp/memory rules do not transfer unchanged to other APIs; exactness must be proved separately | Device properties, sanitizer results, profiler/timing traces, end-to-end output comparison |

No backend may lower precision, truncate coefficients, switch physical models, or activate a fallback to satisfy a capability mismatch. A current browser advertising WebGPU is not proof that the exact requested kernel is supported or correct. A usable graphics backend is not evidence of compute implementation.

Sources: [WebGPU limits and features](https://www.w3.org/TR/webgpu/); [WGSL language and memory model](https://www.w3.org/TR/WGSL/); [Apple GPU feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions); [Apple feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [Apple pipeline execution width](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth); [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
