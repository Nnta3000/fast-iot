#set page(margin: 2.2cm, numbering: "1")
#set text(size: 11pt, lang: "en", region: "gb")
#set par(justify: true)
#set heading(numbering: "1.")

#let thm(title, body) = block(fill: luma(245), inset: 10pt, radius: 4pt, width: 100%)[*#title* \ #body]
#let remark(title, body) = block(stroke: (left: 2pt + luma(150)), inset: (left: 10pt, y: 4pt), width: 100%)[*#title* \ #body]
#let pf(body) = [*Proof.* #body #h(1fr) $square.filled$]
#let algo(title, body) = block(stroke: 0.6pt + luma(110), inset: 10pt, radius: 3pt, width: 100%)[*#title* \ #v(2pt) #body]

#align(center)[
  #text(size: 15pt, weight: "bold")[Hierarchical constraint generation for inverse optimal transport:\ algorithm and correctness]
]

= Overview

*Purpose.* This note is a supplement to the flat proof (_Constraint generation for inverse optimal transport: well-posedness, termination and correctness_). It adds a coarse-to-fine hierarchy to the exact algorithm: a sequence of smaller inverse problems is solved first, and each solution is used to choose the starting edge set of the next finer problem. The idea is borrowed from HELLO (dual-guided hierarchical edge localisation for large-scale OT), with two changes forced by the inverse setting: the unknown $theta$ is level-independent, and a coarse level must be built from the observed support, not from random points.

*Main point.* A level is itself an instance of the flat problem. The flat theory therefore applies to every level unchanged, and the only new facts to prove are:
+ every level satisfies the standing hypotheses of the flat proof (Lemma 1 and Corollary 3);
+ restriction to a sub-grid is a relaxation of the master, which gives monotone bounds and propagates infeasibility (Lemma 2);
+ the starting edge set of a level can be chosen freely, as long as it contains the mandatory set (Lemma 5), so seeding from the coarse solution cannot affect correctness;
+ the output of the hierarchical algorithm equals the output of the flat algorithm (Theorem 6).

*What the proof is not.* It proves termination and correctness. It does _not_ prove that the hierarchy reduces the number of refinement rounds at the fine level. That is a heuristic expectation, and this note contains no experiment. Section 8 lists the open points.

= Setting and imported facts

== Flat instance

A *flat instance* is $Lambda = (I, J, (phi_(i j))_((i,j) in I times J), hat(pi))$ with finite index sets $I, J$ (sources and targets), a feature vector $phi_(i j) in RR^K$ for every pair (computable on demand, never stored as a matrix), and $hat(pi) in RR^(I times J)_(>= 0)$. Its marginals are $a_i = sum_(j in J) hat(pi)_(i j)$ and $b_j = sum_(i in I) hat(pi)_(i j)$, its grid is $U := I times J$ and its support is $S := {(i,j) in U : hat(pi)_(i j) > 0}$. We require $a_i > 0$, $b_j > 0$ and $S != U$. The cost is $c_(i j)(theta) := theta^top phi_(i j)$ for $theta in Theta$, where $Theta subset RR^K$ is a non-empty bounded polytope (for example the simplex) with $0 in.not Theta$.

For a fixed _gauge target_ $j^* in J$ and $S subset.eq E subset.eq U$, the master set is, exactly as in the flat proof,
$ P(E) := { w = (delta, theta, f, g) : theta in Theta, quad g_(j^*) = 0, quad delta >= 0, quad sigma_e (w) = 0 quad forall e in S, quad h_e (w) <= 0 quad forall e in E backslash S }, $
where $sigma_(i j)(w) := f_i + g_j - c_(i j)(theta)$ and $h_e (w) := sigma_e (w) + delta$. Write $P^"full" := P(U)$, $delta^*(E) := max {delta : w in P(E)}$ and $delta^* := delta^*(U)$. (The flat proof fixes the gauge at $g_1 = 0$. Any fixed index works, by renumbering.)

*Remark (what is used about $Theta$ and the cost).* Only three things are used: $theta |-> c_(i j)(theta)$ is linear; $Theta$ is convex, closed and bounded; and (A1) below puts a point of $Theta$ in the argument. Since $Theta$ is a polytope, every $P(E)$ is a polytope in $(delta, theta, f, g)$ and the master is a *linear programme*. The condition $0 in.not Theta$ is not used in any proof: it matters for modelling only, because $theta = 0$ would make every plan optimal (cost identically zero) and give the trivial margin $delta = 0$. The same proofs also hold for any other compact convex $Theta$ (for example a PSD cone slice, where the master becomes an SDP).

== Facts from the flat proof

These are imported without reproof. Each holds for every flat instance.

#thm("Fact F1 (certification; flat Lemma 1).")[
(a) If $w in P^"full"$, then $hat(pi)$ is optimal in $Pi(a,b)$ for the cost $c(theta)$. \
(b) If $hat(pi)$ is optimal for some $theta^* in Theta$ (assumption *(A1)*), then $P^"full" != emptyset$. \
Consequently *(A1) holds if and only if $P^"full" != emptyset$.*
]

#thm("Fact F2 (connectors and compactness; flat Lemmas 2 and 3).")[
(a) Let $K_1, dots, K_p$ be the connected components of the bipartite support graph $G_S$ (all nodes have an edge, because $a, b > 0$). For $p >= 2$ choose a source node $i_k$ and a target node $j_k$ in each $K_k$ and put $R := {(i_k, j_(k+1)) : k = 1, dots, p}$ with $j_(p+1) := j_1$. For $p = 1$ let $R := {e}$ with any $e in U backslash S$. Then $R inter S = emptyset$ and $B := S union R$ satisfies the connector condition (C): the directed graph on the components, with one arc per edge of $B backslash S$, is strongly connected with at least one arc. \
(b) If $B subset.eq E subset.eq U$ and $P(E) != emptyset$, then $P(E)$ is compact (and convex).
]

The flat proof states (b) under (A1); (A1) is used there only to know that $P(E) != emptyset$ (Fact F1(b)). The boundedness argument does not use it.

#thm("Fact F3 (selection rule; flat Lemma 4).")[
Fix rule (L) (lexicographic), or rule (G) (generic direction $r$, outside a null set). If $B subset.eq E$ and $P(E) != emptyset$, there is a unique $v^*(E) in P(E)$ that is maximal for the selection order, and its first coordinate is $delta^*(E)$.
]

*The flat algorithm.* We name the loop of the flat proof so that it can be called on any level.

#algo[FlatRefine($Lambda, F_0; gamma$)][
*Input:* a flat instance $Lambda$ with mandatory set $B$ (Fact F2), a start set $B subset.eq F_0 subset.eq U$, a parameter $gamma >= 1$, and a selection rule (F3). *Output:* INFEASIBLE, or a pair $(v, F)$. \
For $t = 0, 1, dots$:
+ *Solve.* If $P(F_t) = emptyset$, return INFEASIBLE. Otherwise $v_t := v^*(F_t)$.
+ *Detect.* $Q_t := {e in U backslash F_t : h_e (v_t) > 0}$, computed by a _complete_ scan of $U$. If $Q_t = emptyset$, return $(v_t, F_t)$.
+ *Insert.* $V_t :=$ the row-wise and column-wise top-$gamma$ edges of $U backslash F_t$ by score $sigma_e (v_t)$.
+ *Prune.* $T_t := {e in F_t backslash S : h_e (v_t) = 0}$ and $F_(t+1) := B union T_t union V_t$.
]

#thm("Fact F4 (flat Theorem 1, Lemma 5, Theorem 2, Proposition 1).")[
Let $Lambda$ be a flat instance with $P^"full" != emptyset$, and assume exact arithmetic and a rule as in F3. For every $F_0$ with $B subset.eq F_0$: \
(a) FlatRefine stops after at most $2^(|U| - |B|)$ iterations. \
(b) If it returns $(v_T, F_T)$, then $v_T = v^*(U)$. In particular $delta_T = delta^*$, and $hat(pi)$ is optimal for $c(theta_T)$. \
(c) Whenever it does not stop, $v_t succ v_(t+1)$ in the selection order.
]

(In the flat proof the returned point is $v_T$; we additionally return $F_T$ because the hierarchy carries it to the next level.)

= Levels

== Definition

A *level structure* consists of nested index sets
$ I_0 supset.eq I_1 supset.eq dots supset.eq I_L, quad J_0 supset.eq J_1 supset.eq dots supset.eq J_L, $
with $I_0, J_0$ the full source and target index sets of the observed problem. Put $U_ell := I_ell times J_ell$, $S_ell := S inter U_ell$ (the *induced support*: every support edge between retained points is kept), $hat(pi)^ell := hat(pi)|_(U_ell)$, and let $Lambda_ell := (I_ell, J_ell, (phi_(i j))_((i,j) in U_ell), hat(pi)^ell)$. Write $a^ell, b^ell$ for the marginals of $hat(pi)^ell$, $B_ell = S_ell union R_ell$ for the mandatory set of Fact F2 computed on $G_(S_ell)$, and $P_ell (E)$, $P_ell^"full"$, $delta^*_ell (E)$, $delta^*_ell$ for the master objects of $Lambda_ell$. The gauge target is one fixed $j^* in J_L$ (so $j^* in J_ell$ for every $ell$).

Two structural hypotheses are imposed on every level:
- *(H1)* every node of $I_ell$ and of $J_ell$ is an endpoint of an edge of $S_ell$ (hence $a^ell, b^ell > 0$);
- *(H2)* $S_ell != U_ell$.

At level $0$, (H1) and (H2) are the standing assumptions of the flat proof. Under (H1) and (H2), $Lambda_ell$ is a flat instance.

== Construction

#algo[BuildLevels($S; rho, tau, T_"max"$)][
*Input:* the observed support $S$, a sampling ratio $rho in (0,1)$, a grid budget $tau$ and a retry limit $T_"max"$. *Output:* a level structure. \
+ $I_0, J_0 :=$ all indices, $ell := 0$.
+ While $|I_ell| dot |J_ell| > tau$:
  - For $s = 1, dots, T_"max"$: draw $E subset.eq S_ell$ uniformly with $|E| = ceil(rho 2^(-(s-1)) |S_ell|)$; put $I' := {i : (i,j) in E}$, $J' := {j : (i,j) in E}$ and $S' := S inter (I' times J')$. _Accept_ if (i) $|I'| + |J'| < |I_ell| + |J_ell|$ and (ii) $S' != I' times J'$. On acceptance set $I_(ell+1) := I'$, $J_(ell+1) := J'$, $ell := ell + 1$ and leave the loop over $s$.
  - If no draw was accepted, leave the while loop (coarsening has stalled).
+ $L := ell$. Choose $j^* in J_L$.
]

#thm("Lemma 1 (levels are admissible).")[
The output of BuildLevels is a nested level structure that satisfies (H1) and (H2) at every level, and the procedure terminates. Moreover $S_(ell+1) = S_ell inter U_(ell+1)$.
]
#pf[
Nesting holds by construction. Since $S_(ell+1) = S inter U_(ell+1)$ and $U_(ell+1) subset.eq U_ell$, we get $S_ell inter U_(ell+1) = S inter U_ell inter U_(ell+1) = S_(ell+1)$. (H1): each $i in I'$ is the row of some edge in $E subset.eq S'$, and each $j in J'$ is the column of some edge in $E$. (H2) is acceptance test (ii); at level $0$ it is the standing assumption. Each accepted level strictly decreases $|I_ell| + |J_ell|$, and each level uses at most $T_"max"$ draws, so the procedure terminates.
]

#remark[Why edges and not points are sampled.][
Sampling points uniformly can leave a retained point with no support edge, which violates (H1) (then $a^ell_i = 0$ and the instance leaves the flat setting), and it fragments the support graph. Sampling support edges keeps every retained point attached to at least one support edge. Test (ii) is needed because a very small draw can make the induced support the whole grid (for instance a single edge gives a $1 times 1$ grid), in which case the master has no inequality and the margin is unbounded.
]

= Restriction is a relaxation

For $w = (delta, theta, f, g)$ at level $ell$, let $"res"_ell (w) := (delta, theta, f|_(I_(ell+1)), g|_(J_(ell+1)))$.

#thm("Lemma 2 (restriction).")[
For every $ell < L$:
+ $"res"_ell (P_ell^"full") subset.eq P_(ell+1)^"full"$.
+ $delta^*_(ell+1) >= delta^*_ell$, whenever $P_ell^"full" != emptyset$.
+ If $P_(ell+1)^"full" = emptyset$, then $P_ell^"full" = emptyset$.
]
#pf[
(1) Let $w in P_ell^"full"$ and $w' := "res"_ell (w)$. The conditions $theta in Theta$ and $delta >= 0$ are unchanged, and $g_(j^*) = 0$ is kept because $j^* in J_L subset.eq J_(ell+1)$. The equalities required at level $ell + 1$ are those on $S_(ell+1) = S_ell inter U_(ell+1) subset.eq S_ell$, a subset of the level-$ell$ equalities. The margin constraints required at level $ell + 1$ are on $U_(ell+1) backslash S_(ell+1)$. An edge $e$ in this set is in $U_ell$ and not in $S_ell$ (otherwise $e in S_ell inter U_(ell+1) = S_(ell+1)$), so $h_e (w) <= 0$ is a level-$ell$ constraint. The functions $sigma_e, h_e$ for $e in U_(ell+1)$ read only $delta, theta, f_i, g_j$ with $i in I_(ell+1)$, $j in J_(ell+1)$, so $h_e (w') = h_e (w)$. Hence $w' in P_(ell+1)^"full"$.

(2) By Fact F2(b), $P_ell^"full"$ is compact, so it has a maximiser $w$ of $delta$. By (1), $"res"_ell (w) in P_(ell+1)^"full"$ with the same $delta$, so $delta^*_(ell+1) >= delta^*_ell$.

(3) is the contrapositive of the first inclusion: a point of $P_ell^"full"$ would restrict to a point of $P_(ell+1)^"full"$.
]

#remark[Interpretation.][
Lemma 2 is the master-level form of a familiar fact: if $hat(pi)$ is optimal, its restriction to a subset of points (with the induced marginals) is optimal for the restricted problem under the same cost, because the dual potentials restrict. The proof above does not need that statement. It also shows why the coarse support must be the _induced_ support $S inter U_(ell+1)$: a coarse support obtained by sampling edges would treat some tight edges as off-support, which adds margin constraints that the fine level does not have, and the inclusion in (1) would fail.
]

#thm("Corollary 3 (consequences for the hierarchy).")[
Let $(I_ell, J_ell)_(ell = 0)^L$ satisfy (H1), (H2), and let $P_0^"full" != emptyset$ (that is, (A1) holds for the observed problem). Then:
+ $P_ell^"full" != emptyset$ for every $ell$, so every $Lambda_ell$ satisfies (A1), and Facts F1 to F4 apply to every level.
+ $delta^*_L >= delta^*_(L-1) >= dots >= delta^*_0 >= 0$.
+ For every level $ell$ and every $E$ with $B_ell subset.eq E subset.eq U_ell$: $delta^*_ell (E) >= delta^*_ell >= delta^*_0$. In particular, every master value computed at any level, at any iteration, is an upper bound on $delta^*_0$.
]
#pf[
(1) Apply Lemma 2(1) repeatedly: $"res"_(ell-1) compose dots compose "res"_0$ maps $P_0^"full"$ into $P_ell^"full"$, so the latter is non-empty. By Fact F1, $Lambda_ell$ satisfies (A1). (H1), (H2) give the other standing assumptions. (2) is Lemma 2(2) applied repeatedly, and $delta^*_0 >= 0$ because of the constraint $delta >= 0$ and $P_0^"full" != emptyset$. (3) $P_ell^"full" subset.eq P_ell (E)$ gives the first inequality; the second is (2).
]

= Prolongation and seeding

The coarse solution is transferred to the next finer level in two steps: a dual completion (Prolong), then an edge selection (Seed). Both only influence the start set $F_0$, never the problem being solved.

#algo[Prolong($ell; overline(theta), overline(f), overline(g)$)][
*Input:* the parameter and potentials $(overline(theta), overline(f), overline(g))$ of a coarse solution at level $ell + 1$. *Output:* potentials $(tilde(f), tilde(g))$ on $I_ell, J_ell$ and scores $tilde(sigma)_(i j) := tilde(f)_i + tilde(g)_j - c_(i j)(overline(theta))$. \
+ $tilde(f)_i := overline(f)_i$ for $i in I_(ell+1)$ and $tilde(g)_j := overline(g)_j$ for $j in J_(ell+1)$ (retained nodes keep their potentials).
+ For $i in I_ell backslash I_(ell+1)$: $tilde(f)_i := min_(j in J_(ell+1)) (c_(i j)(overline(theta)) - overline(g)_j)$.
+ For $j in J_ell backslash J_(ell+1)$: $tilde(g)_j := min_(i in I_ell) (c_(i j)(overline(theta)) - tilde(f)_i)$.
+ Return $(tilde(f), tilde(g), tilde(sigma))$.
]

The cost is $|I_ell backslash I_(ell+1)| dot |J_(ell+1)| + |J_ell backslash J_(ell+1)| dot |I_ell|$ cost evaluations, as two streamed scans (no cost matrix is stored).

#thm("Lemma 4 (properties of Prolong).")[
Assume $overline(sigma)_e <= 0$ for every $e in U_(ell+1)$, where $overline(sigma)_(i j) = overline(f)_i + overline(g)_j - c_(i j)(overline(theta))$. (This holds for the output $v_T$ of FlatRefine on level $ell + 1$, because $v_T in P_(ell+1)^"full"$.) Then: \
(a) $tilde(sigma)_e = overline(sigma)_e$ for every $e in U_(ell+1)$; in particular $tilde(sigma)_e = 0$ on $S_(ell+1)$. \
(b) $tilde(sigma)_e <= 0$ for every $e in U_ell$; so $(tilde(f), tilde(g))$ is dual feasible for the transport problem of level $ell$ with cost $c(overline(theta))$. \
(c) Every $i in I_ell backslash I_(ell+1)$ has some $j in J_(ell+1)$ with $tilde(sigma)_(i j) = 0$, and every $j in J_ell backslash J_(ell+1)$ has some $i in I_ell$ with $tilde(sigma)_(i j) = 0$.
]
#pf[
(a) holds because retained nodes keep their potentials and the cost is the same function. (b) Let $e = (i,j) in U_ell$. If $i, j$ are both retained, use (a) and the hypothesis. If $i$ is new and $j in J_(ell+1)$, then $tilde(f)_i <= c_(i j)(overline(theta)) - overline(g)_j$ by the minimum in step 2. If $j$ is new, then $tilde(g)_j <= c_(i j)(overline(theta)) - tilde(f)_i$ by the minimum in step 3, which runs over all $i in I_ell$ (all $tilde(f)_i$ are defined by then). If $i$ is retained and $j$ is new, this is the same case. (c) The minimisers in steps 2 and 3 give equality.
]

#remark[What Lemma 4 does not say.][
The prolonged pair is dual feasible, but it is _not_ a point of $P_ell^"full"$: the support edges of $S_ell$ that touch new nodes are generally not tight, and the parameter $overline(theta)$ is the coarse one. The pair is only used to rank candidate edges. The ranking is good when the coarse solution is close to the fine one; there is no guarantee of that (Section 8).
]

#algo[Seed($ell; overline(v), overline(F); kappa$)][
*Input:* the coarse output $overline(v) = (overline(delta), overline(theta), overline(f), overline(g))$ and the coarse final edge set $overline(F) subset.eq U_(ell+1)$, both from FlatRefine on level $ell + 1$; a budget $kappa >= 1$. *Output:* a start set $F_0$ for level $ell$. \
+ $(tilde(f), tilde(g), tilde(sigma)) := "Prolong"(ell; overline(theta), overline(f), overline(g))$.
+ Compute the components of $G_(S_ell)$ and the connector set $R_ell$ of Fact F2(a); $B_ell := S_ell union R_ell$.
+ *Nodewise assignment.* $A_"row" := {(i,j) : i in I_ell, j in "the" kappa "largest" tilde(sigma)_(i j) "among" (i,j) in U_ell backslash S_ell}$, and symmetrically $A_"col"$ with the roles of rows and columns exchanged. (One streamed scan, as in HELLO's dual assignment.)
+ *Carry-over.* $A_"carry" := overline(F) backslash S_(ell+1)$.
+ Return $F_0 := B_ell union A_"row" union A_"col" union A_"carry"$.
]

#thm("Lemma 5 (seeds are admissible).")[
$B_ell subset.eq F_0 subset.eq U_ell$, and $A_"carry" inter S_ell = emptyset$.
]
#pf[
$B_ell subset.eq F_0$ by step 5, and $F_0 subset.eq U_ell$ because $A_"row", A_"col" subset.eq U_ell$ and $overline(F) subset.eq U_(ell+1) subset.eq U_ell$. By Lemma 1, $S_ell inter U_(ell+1) = S_(ell+1)$, so $(overline(F) backslash S_(ell+1)) inter S_ell = emptyset$.
]

The statement of Lemma 5 is the whole correctness requirement on a seed: FlatRefine accepts any $F_0$ with $B_ell subset.eq F_0 subset.eq U_ell$ (Fact F4). The nodewise assignment and the carry-over are chosen for speed only.

= The hierarchical algorithm and its correctness

#algo[HierIOT($(phi_(i j)), hat(pi); rho, tau, T_"max", kappa, gamma$)][
*Input:* observed data and parameters. *Output:* INFEASIBLE, or $v_0 = (delta_0, theta_0, f_0, g_0)$. \
+ $(I_ell, J_ell)_(ell = 0)^L := "BuildLevels"(S; rho, tau, T_"max")$.
+ *Coarsest level.* If $|I_L| dot |J_L| <= tau$, put $F_0^L := U_L$ (dense, so Detect finds nothing); otherwise (coarsening stalled) put $F_0^L := B_L$. Run $"FlatRefine"(Lambda_L, F_0^L; gamma)$ and obtain $(v_L, F_L)$, or return INFEASIBLE.
+ *For $ell = L - 1, dots, 0$:* $F_0^ell := "Seed"(ell; v_(ell+1), F_(ell+1); kappa)$; run $"FlatRefine"(Lambda_ell, F_0^ell; gamma)$ and obtain $(v_ell, F_ell)$, or return INFEASIBLE.
+ Return $v_0$.
]

If $|I_0| dot |J_0| <= tau$ then $L = 0$ and HierIOT is the flat algorithm started from $F_0 = U_0$. Every call to FlatRefine uses a selection rule as in Fact F3; for rule (G), an independent $r^ell tilde N(0, I)$ is drawn for each level (almost surely outside the null set of Fact F3 at that level).

#thm("Theorem 6 (termination and correctness of HierIOT).")[
Assume exact arithmetic, (A1) for the observed problem, and that BuildLevels returns a level structure (it always does, by Lemma 1). Then: \
(a) *Termination.* The call at level $ell$ stops after at most $2^(|U_ell| - |B_ell|)$ iterations. Hence HierIOT stops after at most $sum_(ell = 0)^L 2^(|U_ell| - |B_ell|)$ iterations in total, and never returns INFEASIBLE. \
(b) *Correctness and invariance.* The returned point equals $v_0 = v^*(U_0)$. In particular $delta_0 = delta^*_0$, $hat(pi)$ is optimal for $c(theta_0)$, and $v_0$ is the same point as the output of the flat algorithm with the same selection rule (same $r^0$ for rule (G)), whatever the seeds. \
(c) *Bounds.* At termination of the call at level $ell$, $delta_ell = delta^*_ell$, and $delta^*_L >= dots >= delta^*_0$. Every master value computed during the whole run is at least $delta^*_0$.
]
#pf[
By Corollary 3(1), every $Lambda_ell$ is a flat instance with $P_ell^"full" != emptyset$. At level $L$, $B_L subset.eq F_0^L$ by definition. At level $ell < L$, $B_ell subset.eq F_0^ell$ by Lemma 5. So Fact F4 applies to every call.

(a) Fact F4(a) bounds the iterations of each call. A call returns INFEASIBLE only if $P_ell (F_t) = emptyset$ for some $t$, which is impossible: $P_ell^"full" subset.eq P_ell (F_t)$ and $P_ell^"full" != emptyset$.

(b) By Fact F4(b) at level $0$, the returned point is $v^*(U_0)$, which Fact F3 defines uniquely from the instance and the selection rule alone. The seed $F_0^0$ enters neither. Hence the same point is returned for every seed, in particular for the flat start. Optimality of $hat(pi)$ for $c(theta_0)$ is Fact F1(a), since $v_0 in P_0^"full"$.

(c) Fact F4(b) at level $ell$ gives $delta_ell = delta^*_ell$. The chain is Corollary 3(2) and the last claim is Corollary 3(3).
]

#remark[What the hierarchy changes, and what it does not.][
By Theorem 6(b), the hierarchy cannot change the answer: only the path to it. The coarse levels contribute (i) a start set $F_0^0$ that, one hopes, makes the fine level stop after fewer rounds, and (ii) the bounds of Theorem 6(c) and the infeasibility test of Corollary 8. The _exactness_ of the final certificate comes only from the complete scan at level $0$.
]

= Variants

#thm("Corollary 7 (grow-only mode: no selection rule needed).")[
Replace the Prune step of FlatRefine by $F_(t+1) := F_t union V_t$, and let $v_t$ be _any_ maximiser of $delta$ over $P(F_t)$ (no tie-break). Then for every $F_0$ with $B subset.eq F_0$, FlatRefine stops after at most $|U| - |F_0|$ iterations. If it returns $(v_T, F_T)$, then $v_T in P^"full"$ and $delta_T = delta^*$. In particular, Theorem 6(a) and (c) hold for HierIOT in this mode, with the bound $sum_ell (|U_ell| - |F_0^ell|)$, and the output of level $0$ is a maximiser of $delta$ on $P_0^"full"$ (not necessarily independent of the seeds).
]
#pf[
If $Q_t != emptyset$, the set $V_t$ contains an edge of $Q_t$ and is disjoint from $F_t$ (property (V) of the flat proof, which holds for the top-$gamma$ rule). Hence $|F_t|$ strictly increases and is bounded by $|U|$. At the stop, $h_e (v_T) <= 0$ for all $e in U backslash F_T$, and $h_e (v_T) <= 0$ on $F_T backslash S$ by feasibility, so $v_T in P^"full"$. Since $P^"full" subset.eq P(F_T)$, we get $delta_T = max_(P(F_T)) delta >= max_(P^"full") delta >= delta_T$. The emptiness argument of Theorem 6(a) is unchanged.
]

Grow-only mode loses the memory control of pruning (the edge set never shrinks), but its termination argument is trivial, and it needs no second solve for a tie-break.

#thm("Corollary 8 (infeasibility test and margin test, without (A1)).")[
Do not assume (A1), and run HierIOT with the convention that a call returns INFEASIBLE when $P_ell (F_t) = emptyset$ at some Solve step. \
(a) If HierIOT returns INFEASIBLE, then (A1) fails: no $theta in Theta$ makes $hat(pi)$ optimal. \
(b) If HierIOT returns $v_0$, then (A1) holds, and Theorem 6(b), (c) hold. \
(c) Let $delta_"min" >= 0$. If at any level and any iteration the solved master value $delta^*(F_t)$ is below $delta_"min"$, then $delta^*_0 < delta_"min"$: no solution of the observed problem has margin $delta_"min"$. The run may stop there.
]
#pf[
(a) Suppose (A1) held. Then $P_0^"full" != emptyset$ by Fact F1(b), and by Corollary 3(1) $P_ell^"full" != emptyset$ for all $ell$. Then $P_ell (F_t) supset.eq P_ell^"full"$ is never empty, a contradiction. (b) A returned $v_0$ lies in $P_0^"full"$, so (A1) holds (Fact F1(a)). The proofs of Facts F3 and F4 use (A1) only to know that the relevant sets $P(E)$ are non-empty (re-reading them: compactness, the unique maximiser, and the strict decrease all take place inside non-empty sets, and emptiness is detected by the Solve step), so they apply. (c) By Corollary 3(3), $delta^*(F_t) >= delta^*_ell >= delta^*_0$.
]

Corollary 8(a) is one-sided: a coarse level can be feasible while the full problem is not, so a feasible coarse level proves nothing about the fine level.

#remark[The relaxed (regret-based) variant.][
BuildLevels, Prolong, Seed and the scans carry over unchanged. For the relaxed master (inequality on all of $F$, objective the regret) one must recompute the induced weights $a^ell, b^ell$ and the induced average feature $overline(phi)^ell$ from $hat(pi)^ell$ at each level. There is no analogue of Lemma 2 for the regret, and no finite-termination theorem, so there the coarse levels are warm starts only, with no bounds.
]

= What is not proved

- *No speed-up guarantee.* Nothing here bounds the number of fine-level rounds. The seed is a heuristic: the coarse $overline(theta)$ solves a problem with fewer constraints, so it is less determined than the fine optimum, and the master returns a particular maximiser selected by rule (L) or (G), not a "central" one.
- *No transfer of HELLO's rates.* HELLO's dual-inheritance bound concerns randomly subsampled _marginals_ of a latent distribution. In the inverse problem the coarse level is an induced sub-problem of an observed plan, and no analogous bound is claimed.
- *Level construction.* Lemma 1 proves that BuildLevels terminates and yields admissible levels, but not that it reaches the budget $tau$ (coarsening can stall; the algorithm then starts the coarsest level from $B_L$), nor how fast the grids shrink. Edge sampling does not give a uniform sample of points.
- *Pruning variants.* The proof covers strict pruning (keep $B union T_t union V_t$) with a selection rule, and grow-only mode. It does not cover hysteresis pruning, one-solve perturbed rules, or HELLO's budgeted pruning to a fixed size $O(m + n)$: dropping non-active edges to meet a budget is not an application of flat Theorem 1, and the strict-decrease argument does not apply to it.
- *Cost.* Each round at level $ell$ scans all of $U_ell$, which costs $|U_ell|$ cost evaluations. The saving is only in the number of rounds at the finest level, and is empirical. At a few hundred points there is little to save.
- *Floating point.* As in the flat proof, all statements are for exact arithmetic with exact tests. Prolong is exact in the sense of Lemma 4 only up to rounding; this affects the quality of the seed, never correctness.
- *Noisy plans.* Without (A1) the exact master is generically infeasible. Corollary 8 gives a cheaper way to detect that, not a way to solve it.
- *No experiment.* This note contains no numerical evidence for the hierarchy. The first test to run: the flat algorithm against HierIOT on a clean instance with $n$ in the thousands, recording the number of rounds and the scan count at level $0$, with Theorem 6(b) (identical output) as a built-in correctness check.
