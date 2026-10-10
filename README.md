# Roses.cpp

C++ port of [Roses.py](../Roses.py) — animated 8-particle rose
attractor with CRT phosphor trails.

**R1 (current): single-threaded.** `roses.cpp` is pure C++17
standard library, no third-party dependencies. It contains all
attractor math, density planes, tone-mapping, and rose history,
and speaks a tiny pipe protocol (frames out on stdout, one-letter
commands in on stdin). `display.py` is a thin tkinter shim that
launches the binary, displays its frames, and forwards keypresses
— the only Python in the repo, because C++ has no standard GUI.

## Build & run

```sh
g++ -O2 -std=c++17 -o roses roses.cpp
python3 display.py
```

Controls are identical to roses.py: Space next rose, Backspace
previous, p phosphor toggle, `-`/`=` persistence, Enter/Escape
fullscreen, q quit.

## Roadmap

- **R1** single-threaded core + display shim (this check-in)
- **R2** multithreaded: one `std::thread` per particle. Density is
  already stored plane-major (plane j at offset `j*w*h`) so each
  thread owns a disjoint plane — no locks, no false sharing.
