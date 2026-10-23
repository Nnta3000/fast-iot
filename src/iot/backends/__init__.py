"""
CN: **可替换的后端**：LP 引擎（CuPDLPx）与分离扫描（torch 通用 / HELLO 专用），以及双线性嵌入。
    这些是实现选择，不是算法选择。
EN: **Swappable backends**: the LP engine (CuPDLPx) and the separation scans (generic torch / HELLO),
    plus the bilinear embedding they rely on. Implementation choices, not algorithm choices.
"""

from .bilinear import BilinearEmbeddable, BilinearEmbedding, bilinear_embedding, features_support_bilinear
from .cupdlpx_master import CuPdlpxLp, CuPdlpxMasterSolver, build_cupdlpx_lp
from .generic_scan import scan_violations_torch
from .hello_scan import fast_scan_supported, scan_violations_fast
from .torch_runtime import cuda_available, describe_backend, resolve_torch_device, torch_available

__all__ = [
    "BilinearEmbeddable", "BilinearEmbedding", "CuPdlpxLp", "CuPdlpxMasterSolver",
    "bilinear_embedding", "build_cupdlpx_lp", "cuda_available", "describe_backend",
    "fast_scan_supported", "features_support_bilinear", "resolve_torch_device",
    "scan_violations_fast", "scan_violations_torch", "torch_available",
]
