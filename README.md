# Roses.cpp

C++ port of [Roses.py](https://github.com/davec95/Roses.py) —
animated 8-particle rose attractor with CRT phosphor trails.

**R3 (current): native GUI + iteration report.** `roses.cpp` is
C++20 standard library plus the system X11 libraries (Xlib +
MIT-SHM, no third-party deps). Run `./roses` in a terminal and it
opens its own window and renders straight into it — no Python
needed. On exit it prints iterations/sec to stderr (1 iteration =
1 frame = one full pass over all 8 color planes).

`display.py` (tkinter) is now optional legacy: `./roses` piped
still speaks the R1/R2 PPM protocol, so
`./roses | python3 display.py --stdin` still works.

## Build & run

```sh
make
./roses          # native X11 window (needs DISPLAY)
```

Controls are identical either way: Space next rose, Backspace
previous, p phosphor toggle, f/F shading toggle (off = every hit
pixel at full brightness), `-`/`=` persistence, Enter/Escape
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
- **R2** multithreaded: 8 workers, one per attractor
- **R2.1** f/F shading toggle (flat = every hit at full brightness)
- **R3** native X11 GUI mode (MIT-SHM blit, no shim) + iterations/sec
  report on exit (this check-in)
