# warp_backend — GPU training backend (GEM-validated)

Batched BLDC physics (Warp, float64) + batched torch actors + NumPy
per-env encoders, behind the identical 7-observation outer contract.
GEM CPU remains the validation judge.

Parity gates (all green): RHS f64 1e-13, rollout 5e-13, full-step open
loop <=2%, PI closed loop omega 6e-4, trip step within +-5 physics steps (test_trip.py),
backend-vs-OuterEnv
obs 2.4e-5, selected-actor scores within 0.2%.

GEM pieces resolve via `third_party/gym-electric-motor` when it carries
them, else `$VROOM_GEM_PATH`, else the documented working tree.
