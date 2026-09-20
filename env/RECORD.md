# Linux env record (2026-09-20)

- Manager: `uv 0.11.28`, `uv venv --python 3.12`, pins per `requirements.txt` (match TRANSFER source: torch 2.13.0, SB3 2.9.0, gymnasium 1.3.0, numpy 2.5.2).
- torch `2.13.0+cu130`, `cuda.is_available()=True` (RTX 3060 12GB, driver 580, CUDA 13.0).
- GEM installed editable from `third_party/gym-electric-motor` (submodule master@2544f13).
- Box: 24 threads, 15GB RAM, 37G free on / (92% used — watch disk).
- Microbench (Pendulum, DDPG 2x128/2x256, 4000 steps): cpu 21.5s, cuda 8.0s (~2.7x on updates).
