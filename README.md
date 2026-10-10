# Roses.cpp

C++ port of [Roses.py](https://github.com/davec95/Roses.py) —
animated 8-particle rose attractor with CRT phosphor trails.

**R2 (current): multithreaded.** `roses.cpp` is pure C++20
standard library, no third-party dependencies. It contains all
attractor math, density planes, tone-mapping, and rose history,
and speaks a tiny pipe protocol (frames out on stdout, one-letter
commands in on stdin). `display.py` is a thin tkinter shim that
launches the binary, displays its frames, and forwards keypresses
— the only Python in the repo, because C++ has no standard GUI.

## Build & run

```sh
make
python3 display.py
```

Controls are identical to roses.py: Space next rose, Backspace
previous, p phosphor toggle, `-`/`=` persistence, Enter/Escape
fullscreen, q quit.

## Threading model (R2)

One worker thread per attractor: worker j owns particle j and
density plane j outright (plane-major layout, disjoint memory),
so the hot loop has no mutexes and no atomics. Main + 8 workers
pass three `std::barrier`s per tick: step-start → phase 1 (each
worker decays + steps its own plane) → step-done → phase 2 (all
9 threads tone-map disjoint pixel slices) → tone-done → main
emits the frame. Commands, resizes, and reseeds happen in the
main-only window while workers are parked at step-start.

## History

- **R1** single-threaded core + display shim
- **R1.1** TTY guard on direct `./roses` runs
- **R2** multithreaded: 8 workers, one per attractor (this check-in)
