"""
EN: **EOT -- the entropic inverse-OT master.** Minimise, on an edge set `F`,

        `Psi_F(theta, f, g) = <phi_bar, theta> - a.f - b.g + eps * sum_{e in F} exp(sigma_e / eps)`,
        `sigma_e = f_i + g_j - theta.phi_e`,

    instead of the max-margin (`delta`) of the exact loop. `Psi_F` is convex in `(theta, f, g)`.

    | module | what it is | is it "EOT proper"? |
    | --- | --- | --- |
    | `joint_master.py` | Newton-CG on `(theta, g)`, `f` eliminated in closed form; ONE solve | **yes** |
    | `joint_reference.py` | independent reduced-objective SLSQP (scipy `logsumexp`), the trusted reference | reference |
    | `soft_master.py` | NESTED form: SLSQP on `theta` with an entropic OT solve per `theta` | no -- comparison only |
    | `inner_f.py` | the inner `(f, g)` solve on a sparse edge set (Newton or Sinkhorn) | component |
    | `soft_loop.py` | the EOT loop: soft master + separation scan + prune + edge advance | driver |

    Studies live in `iot_experiments/studies/real/t1t2/metric_learning/eot_*.py`; reports in
    `iot_experiments/reports/current/eot/`.

    CN: **EOT —— 熵正则反 OT 的联合 master。** 在边集 `F` 上最小化 `Ψ_F`（见上），而不是精确循环里的
    最大间隔 `delta`。`Ψ_F` 对 `(θ,f,g)` 凸。哪些模块是"EOT 本体"见上表；`soft_master.py` 是嵌套形式，
    只用于比较，**不是** EOT 本体。
"""
