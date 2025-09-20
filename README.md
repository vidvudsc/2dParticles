# Barnes–Hut N-Body Simulation (with Collisions)

A 2D particle-based galaxy-style simulation written in **C** with [raylib](https://www.raylib.com/).  
Supports **Barnes–Hut gravity**, **always-on elastic collisions**, and multiple **view modes**.

---

## Features
- Barnes–Hut gravity tree with adjustable opening angle (`[` and `]`)
- Adaptive timestep for stability in dense regions
- Always-on elastic collisions (particles push apart, no merging)
- Bodies spawn in a circular disk with different velocity bands (galaxy-like structure)
- View modes (`V` key):
  - Speed coloring
  - Mass coloring
  - Acceleration coloring
  - Density coloring
  - Random colors
- Barnes–Hut tree visualization (`B` key)
- Real-time HUD with FPS, timestep, particle count, etc.

<img width="1284" height="750" alt="Screenshot 2025-09-20 at 23 05 15" src="https://github.com/user-attachments/assets/df88c802-e5e4-4b1d-a925-d6b7176d513c" />


<img width="1278" height="749" alt="Screenshot 2025-09-20 at 23 04 47" src="https://github.com/user-attachments/assets/8db82ce8-bc3b-4c27-98ff-36149ac3e52e" />

---

## Controls
- **Arrow Keys** → Move camera
- **W / S** → Zoom in/out
- **Space** → Pause/resume
- **R** → Reset bodies
- **V** → Cycle view modes
- **B** → Toggle Barnes–Hut tree visualization
- **[ / ]** → Decrease/increase Barnes–Hut opening angle

---

## Build (macOS with Homebrew raylib)

### Single-threaded (simple)
```bash
gcc nbody_barnes_hut_perf_views_omp.c -o nbody -O3 -Ofast -ffast-math -march=native -Wall -Wextra -std=c11 \
  -I/opt/homebrew/include -L/opt/homebrew/lib \
  -lraylib -framework Cocoa -framework IOKit -framework OpenGL -lm
