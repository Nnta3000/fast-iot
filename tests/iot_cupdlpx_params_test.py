"""
EN: The CuPDLPx parameter whitelist must equal the API's own parameter names.

    This guards a bug that was silent for days: the whitelist had been written from memory (`iter_limit`,
    `eps_opt`, `eps_feas`, `eps_infeas_detect` -- none of which exist), and the binding **ignores unknown
    keys**, so `iteration_limit=5` was accepted by our check and then dropped by the solver. Every "tuned"
    run measured nothing. The real names live in `pycupdlpx.make_default_params()`.

CN: CuPDLPx 参数白名单必须与 API 自身的参数名完全一致。此前名单是凭印象写的，而 binding **静默忽略未知键**，
    于是"调参"从来没生效过。真实名字在 `pycupdlpx.make_default_params()` 里。
"""

from __future__ import annotations

import pytest


def test_cupdlpx_whitelist_matches_the_api() -> None:
    """CN: 名单与 API 必须逐键相同（多一个或少一个都算回归）。EN: the set must equal the API's, key for key."""
    torch = pytest.importorskip("torch")  # noqa: F841 - pycupdlpx needs torch's shared libs loaded first
    pycupdlpx = pytest.importorskip("pycupdlpx")
    from iot.backends.lp_common import CUPDLPX_KNOWN_PARAMS

    api = set(pycupdlpx.make_default_params())
    missing = sorted(api - set(CUPDLPX_KNOWN_PARAMS))
    extra = sorted(set(CUPDLPX_KNOWN_PARAMS) - api)
    assert not missing, f"parameter names the API accepts but the whitelist rejects: {missing}"
    assert not extra, f"parameter names the whitelist allows but the API does not know: {extra}"


def test_iteration_limit_is_in_the_whitelist() -> None:
    """CN: 迭代上限的真实名字是 `iteration_limit`。EN: the iteration cap's real name is `iteration_limit`."""
    from iot.backends.lp_common import CUPDLPX_KNOWN_PARAMS

    assert "iteration_limit" in CUPDLPX_KNOWN_PARAMS
    assert "iter_limit" not in CUPDLPX_KNOWN_PARAMS
