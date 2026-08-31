# Final Method

## Problem Setting

Consider `N` quadrotors with positions `p_i`, velocities `v_i`, and a shared goal region. Each agent receives local self/neighbor observations and outputs four normalized motor commands. Obstacles and other agents make a direct goal vector insufficient: independent policies can choose geometrically invalid routes, while successful agents can remain in the bottleneck and block later arrivals.

The final method separates global liveness coordination from bounded low-level control:

```text
physical state + obstacles
        -> A* route and visible waypoint
        -> synchronized stage/enter/goal/egress state
        -> waypoint-conditioned observation
        -> one bounded BC controller
        -> four motor commands
```

There is no online expert pool, policy averaging, learned routing gate, inactive shield, or DAgger module in the released method.

## Obstacle-Aware Route

Obstacle geometry is inflated by a fixed clearance buffer `b = 0.35 m`. For a grid edge `(x,y)`, the planner uses

```text
c(x,y) = ||x-y||_2,
```

and computes the minimum-cost 8-connected A* path on a `0.25 m` grid. Let the raw path be

```text
P_i = (q_i^0, q_i^1, ..., q_i^K).
```

Visibility compression removes intermediate nodes only when the full connecting segment is collision-free in the inflated geometry. The active waypoint is

```text
w_i(t) = first unreached visible waypoint in P_i(t).
```

Routes are recomputed every 25 frames or when invalidated. A waypoint is reached inside `0.30 m`. This produces short obstacle-feasible targets without asking the neural controller to infer global geometry from local observations.

## Synchronized Stage-Enter-Egress Coordinator

Each agent follows a deterministic finite-state process:

```text
route -> stage -> enter -> goal dwell -> egress.
```

The coordinator first assigns staging targets outside the conflict region. Entry is released only after the readiness condition is satisfied:

```text
R(t) = AND_i [ ||p_i(t)-s_i||_2 <= r_ready ],
```

subject to the frozen maximum staging duration. Deterministic ordering resolves simultaneous candidates. After goal dwell, an outward egress target is assigned so that a completed vehicle does not occupy the shared-goal bottleneck.

The coordinator modifies target waypoints only. It never combines motor actions from different learned policies.

## Analytic Teacher

For active target `p_i*`, the teacher forms desired translational acceleration

```text
a_i* = K_p (p_i* - p_i) - K_d v_i + g e_3,
```

with `K_p = 4.5` and `K_d = 3.5`. The desired thrust direction follows from `a_i*`. Geometric attitude feedback uses proportional and derivative gains `200` and `50`, with yaw error scaled by `0.2`. Collective thrust and body moments are mapped to four motor commands through the inverse simulator motor Jacobian and clipped to the simulator action range.

The teacher is used only to generate labels; it is not called by the deployed neural policy.

## Waypoint-Conditioned Observation

Let the original local observation be `o_i`. Goal-relative fields are replaced by active-waypoint displacement:

```text
Delta w_i(t) = w_i(t) - p_i(t).
```

All self-state and neighbor features remain unchanged. The transformed observation `o_i^w` therefore exposes a local obstacle-feasible control target while preserving the original simulator interface.

## Bounded Neural Controller

For normalized input

```text
z_i = (o_i^w - mu) / (sigma + epsilon),
```

the controller is

```text
h_1 = SiLU(W_1 z_i + b_1),
h_2 = SiLU(W_2 h_1 + b_2),
h_3 = SiLU(W_3 h_2 + b_3),
u_i = tanh(W_4 h_3 + b_4).
```

Hidden widths are `256`, `256`, and `128`. The four outputs satisfy `u_i in [-1,1]^4`. Observation statistics `mu` and `sigma` are stored in each checkpoint.

Given teacher actions `u_n^T`, behavioral cloning minimizes

```text
L_BC(theta) = (1/M) sum_n || pi_theta(o_n^w) - u_n^T ||_2^2.
```

The final dataset has 179,456 training labels from seeds `160000..160031` and 22,432 validation labels from seeds `161000..161003`. Adam uses learning rate `1e-3`, batch size `4096`, and 60 epochs. The three formal training seeds are `171001`, `171002`, and `171003`.

## Online Algorithm

For each simulator frame:

1. Read physical state and obstacle geometry.
2. Replan an agent route when the 25-frame condition or invalidation condition fires.
3. Update the synchronized stage-enter-goal-egress state machine.
4. Select the active route, staging, goal, or egress waypoint.
5. Replace goal-relative observation fields with waypoint displacement.
6. Evaluate the single bounded controller once per agent.
7. Apply the four motor commands and update reach/dwell state.

The planner and coordinator handle route feasibility and bottleneck liveness; the bounded controller handles continuous dynamics. This division is the central implemented design, not an auxiliary fallback.

## Computational Cost

A* is refreshed intermittently, while coordination and neural inference execute every frame. On the isolated RTX 5090 benchmark, the grand means are `0.904 ms/frame` for policy inference, `2.405 ms/frame` for route/coordination logic, and `9.020 ms/frame` end to end.

## Component Evidence

The released component study is in `results/final_component_ablation/`. Obstacle waypoints provide the largest collision reduction; synchronized coordination removes bottleneck conflict while preserving completion; bounded BC maintains those outcomes with lower policy latency. The DAgger candidate changed no terminal outcome and was therefore excluded from the final method.
