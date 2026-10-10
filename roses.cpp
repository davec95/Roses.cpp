/*
 * roses.cpp - multithreaded C++ port of Roses.py/roses.py (R2)
 *
 * Animated 8-particle rose (Clifford-style) attractor with CRT
 * phosphor trails:
 *
 *     x' = a*x + y + c
 *     c' = 2*b*x'^2 / (1 + x'^2)
 *     y' = a*x' - x + c'
 *
 * Eight particles each accumulate hit DENSITY into their own
 * float plane that behaves like the phosphor coating of an old
 * CRT: every tick each plane decays by the persistence factor
 * and new hits are added to that particle's plane. Each plane
 * is tone-mapped separately, then the 8 colors are composited
 * additively. A pixel visited only by the red particle
 * saturates toward red, never toward white -- white appears
 * only where different colors genuinely overlap.
 *
 * Pure C++20 standard library, no third-party dependencies.
 * This file is the compute core only; it renders frames and
 * speaks a tiny pipe protocol; display.py (tkinter) is the
 * display shim that launches this binary, shows its frames,
 * and forwards keypresses back as commands.
 *
 * ---------------------------------------------------------------
 * THREADING MODEL (R2): one thread per attractor
 *
 * The 8 particles never interact: particle j writes only plane
 * j. So worker thread j owns particle j AND density plane j
 * outright -- the hot loop needs no mutexes, no atomics, and
 * no false sharing (plane-major layout: plane j at offset
 * j*w*h, disjoint memory).
 *
 * Per tick, 8 workers + the main thread pass three barriers:
 *
 *   [step-start]  main pumped stdin and applied commands in the
 *                 main-only window before this barrier: resize,
 *                 reseed, phosphor toggle all touch shared state
 *                 only while workers are parked at step-start.
 *       |
 *   PHASE 1: worker j decays plane j, then steps particle j
 *            STEPS_PER_TICK times, accumulating directly into
 *            plane j. Main idles.
 *       |
 *   [step-done]
 *       |
 *   PHASE 2: all 9 threads tone-map a disjoint slice of pixels
 *            (read all planes, write only their own rgb slice).
 *       |
 *   [tone-done] main emits the frame, loops.
 *
 * std::barrier arrive_and_wait() provides happens-before edges,
 * so plain (non-atomic) shared data is race-free when written
 * strictly between two barriers. a/b/phosphor/decay_i are
 * atomics only as cheap insurance for cross-thread visibility.
 *
 * ---------------------------------------------------------------
 * PROTOCOL (core side)
 *
 *   stdout, one frame at a time:
 *       F <w> <h> <title>\n          frame descriptor line
 *       P6\n<w> <h>\n255\n           PPM header
 *       <w*h*3 raw RGB bytes>        pixel data
 *
 *   stdin, one command per line:
 *       n            next rose (replay forward, or new random
 *                    rose when the cursor is already newest)
 *       b            previous rose (walk cursor back)
 *       p            toggle phosphor trails (off = solid dots)
 *       f            toggle flat shading (on = every hit pixel
 *                    lands at full color/brightness, no glow
 *                    tone-curve and no phosphor fade)
 *       -            phosphor shorter persistence
 *       =            phosphor longer persistence
 *       r <w> <h>    set framebuffer size (ignored if unchanged)
 *       R            reseed particles (render)
 *       q            quit
 *
 * ---------------------------------------------------------------
 * DENSITY LAYOUT
 *
 * PLANE-MAJOR: plane j occupies density[j*w*h .. (j+1)*w*h).
 * Hits accumulate with a direct `plane[idx] += 1.0f` -- exact
 * for repeated hits, no numpy bincount workaround needed. The
 * layout also gives each worker thread a disjoint memory region.
 *
 * Build:  make            (or: g++ -O2 -std=c++20 -pthread -o roses roses.cpp)
 * Run:    python3 display.py        (launches ./roses itself)
 */

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <deque>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <poll.h>
#include <unistd.h>

// --- window / attractor constants --------------------------------

static const int   WINDOW_W = 1280;
static const int   WINDOW_H = 700;
static const double SCALE   = 9.5;

static const int BUFFER_SIZE = 20;

// --- phosphor settings -------------------------------------------

static const float PHOSPHOR_GAIN = 1.0f;   // density one hit adds
static const float GLOW_K = 50.0f;         // tone-map knee; lower = hotter glow

static const float DECAY_STEPS[] = {0.90f, 0.94f, 0.96f, 0.98f, 0.99f};
static const int   N_DECAY = (int)(sizeof(DECAY_STEPS) / sizeof(DECAY_STEPS[0]));

// EGA palette 8..15, normalized 0..1 for compositing
static const int N_COLORS = 8;
static const float COLOR_NORM[N_COLORS][3] = {
    { 85.0f/255,  85.0f/255,  85.0f/255 },   // #555555
    { 85.0f/255,  85.0f/255, 255.0f/255 },   // #5555ff
    { 85.0f/255, 255.0f/255,  85.0f/255 },   // #55ff55
    { 85.0f/255, 255.0f/255, 255.0f/255 },   // #55ffff
    {255.0f/255,  85.0f/255,  85.0f/255 },   // #ff5555
    {255.0f/255,  85.0f/255, 255.0f/255 },   // #ff55ff
    {255.0f/255, 255.0f/255,  85.0f/255 },   // #ffff55
    {255.0f/255, 255.0f/255, 255.0f/255 },   // #ffffff
};

static const int STEPS_PER_TICK = 20000;

// --- shared state -------------------------------------------------

// density is PLANE-MAJOR: plane j occupies
// density[j*w*h .. (j+1)*w*h). See header comment.
// Reallocated by main only, while workers are parked at
// step-start (the main-only window).
static std::vector<float>   density;
static std::vector<uint8_t> rgb;          // tone-mapped frame
static int fb_w = 0;                      // frozen between barriers
static int fb_h = 0;

struct Particle { double x, y, c; };
// particles[j]: written by main in render() (main-only window),
// then read/written only by worker j during phase 1.
static Particle particles[N_COLORS];

// atomics: cheap insurance for cross-thread visibility; the
// barriers already order the writes
static std::atomic<double> a{0.0};
static std::atomic<double> b{0.0};
static std::atomic<bool>   phosphor{true};
static std::atomic<int>    decay_i{2};

// SHADING: the CRT look as a whole -- the glow tone-curve that
// shades a pixel's brightness by its accumulated density, plus
// the phosphor fade that dims old hits. When shading is OFF
// (key f/F) neither applies: the plane is cleared each tick and
// every pixel hit this tick lands at FULL color/brightness.
static std::atomic<bool>   shading{true};

static std::deque<std::pair<double,double>> ab_buffer;  // history (main only)
static int index = 0;                                    // cursor (main only)

static std::atomic<bool> quit_requested{false};

// three barriers: main + 8 workers
static std::barrier step_start{N_COLORS + 1};
static std::barrier step_done{N_COLORS + 1};
static std::barrier tone_done{N_COLORS + 1};

// --- random -------------------------------------------------------

/*
 * time_random: uniform(lo, hi) with the current time folded into
 * the seed, mirroring the Python original -- wall clock in
 * nanoseconds XOR 64 bits of OS entropy, so two roses drawn at
 * different instants essentially never share the same (a, b).
 * Main thread only.
 */
static double time_random(double lo, double hi)
{
    static std::random_device rd;

    uint64_t ns = (uint64_t)
        std::chrono::system_clock::now()
            .time_since_epoch().count();

    uint64_t seed = ns ^ ((uint64_t)rd() << 32) ^ (uint64_t)rd();

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

// --- accumulator / render (main thread, main-only windows) --------

static void create_accumulator(int w, int h)
{
    fb_w = w;
    fb_h = h;
    density.assign((size_t)N_COLORS * w * h, 0.0f);
    rgb.assign((size_t)w * h * 3, 0);
}

static void render(void)
{
    // reseed the 8 particles along the diagonal, as in the original
    double cb = b.load();
    for (int i = 0; i < N_COLORS; i++) {
        double t = 2.0 * i - 0.51;
        double x = t;
        double y = t;
        double c = 2.0 * cb * x * x / (1.0 + x * x);
        particles[i] = { x, y, c };
    }
}

// --- rose history (same sliding-window semantics as roses.py) -----
// main thread only, in the main-only window between barriers

static void next_rose(void)
{
    double ca = a.load(), cb = b.load();

    if (index < (int)ab_buffer.size() - 1) {
        // cursor not at newest: replay the buffered rose exactly
        index++;
        ca = ab_buffer[index].first;
        cb = ab_buffer[index].second;
    } else {
        // cursor at newest: THIS is where a new random rose is allowed
        ca = time_random(-1.0, 1.0);
        cb = 1.0 - ca;

        ab_buffer.push_back({ ca, cb });
        if ((int)ab_buffer.size() > BUFFER_SIZE)
            ab_buffer.pop_front();      // window full: drop oldest

        index = (int)ab_buffer.size() - 1;
    }

    a.store(ca);
    b.store(cb);
    render();
}

static void back_rose(void)
{
    if (index > 0) {
        index--;
        a.store(ab_buffer[index].first);
        b.store(ab_buffer[index].second);
    }
    render();
}

static void toggle_phosphor(void)
{
    phosphor.store(!phosphor.load());
    // clear trails: safe here, workers are parked at step-start
    std::fill(density.begin(), density.end(), 0.0f);
}

static void toggle_shading(void)
{
    shading.store(!shading.load());
    // clear planes so stale faded pixels don't linger under the
    // new mode: safe here, workers are parked at step-start
    std::fill(density.begin(), density.end(), 0.0f);
}

static void shorter_persistence(void)
{
    decay_i.store(std::max(0, decay_i.load() - 1));
}

static void longer_persistence(void)
{
    decay_i.store(std::min(N_DECAY - 1, decay_i.load() + 1));
}

// --- tone-map: one disjoint pixel range per thread ----------------

/*
 * Tone-map pixels [lo, hi): read all 8 planes, write only this
 * range's rgb bytes. Empty pixels (d == 0) skip the expf
 * entirely -- most pixels are empty, a large constant-factor win.
 * Called by main (slice 0) and by every worker (slice j+1) in
 * phase 2, after step_done guarantees all planes are final.
 */
static void tone_map_range(size_t lo, size_t hi, size_t NP)
{
    const bool ph = phosphor.load();
    const bool sh = shading.load();

    for (size_t p = lo; p < hi; p++) {
        float r = 0.0f, g = 0.0f, bl = 0.0f;

        for (int k = 0; k < N_COLORS; k++) {
            float d = density[(size_t)k * NP + p];
            if (d <= 0.0f) continue;

            float inten;
            if (!sh)
                inten = 255.0f;               // flat: any hit = full color
            else if (ph)
                inten = 255.0f * (1.0f - expf(-d / GLOW_K));
            else
                inten = fminf(d, 1.0f) * 255.0f;

            r  += inten * COLOR_NORM[k][0];
            g  += inten * COLOR_NORM[k][1];
            bl += inten * COLOR_NORM[k][2];
        }

        rgb[3*p    ] = (uint8_t)std::min(255.0f, std::max(0.0f, r));
        rgb[3*p + 1] = (uint8_t)std::min(255.0f, std::max(0.0f, g));
        rgb[3*p + 2] = (uint8_t)std::min(255.0f, std::max(0.0f, bl));
    }
}

// --- worker: one thread per attractor -----------------------------

/*
 * Worker j: decay plane j, step particle j, then tone-map pixel
 * slice j+1. Plane j and particle j are touched by nobody else;
 * rgb slices are disjoint. No locks, no atomics in the hot loop.
 */
static void worker(int j)
{
    const double sc = SCALE;

    for (;;) {
        step_start.arrive_and_wait();
        if (quit_requested.load()) break;

        const size_t NP = (size_t)fb_w * fb_h;   // fb_* frozen between barriers
        const int cx = fb_w / 2;
        const int cy = fb_h / 2;
        const double ca = a.load();
        const double cb = b.load();
        float* plane = density.data() + (size_t)j * NP;

        // phosphor decay: this plane dims a little each tick.
        // With shading off there is no fade: clear the plane so
        // only this tick's hits show, at full brightness.
        if (!shading.load()) {
            std::fill(plane, plane + NP, 0.0f);
        } else if (phosphor.load()) {
            const float k = DECAY_STEPS[decay_i.load()];
            for (size_t i = 0; i < NP; i++)
                plane[i] *= k;
        } else {
            std::fill(plane, plane + NP, 0.0f);
        }

        // step particle j and accumulate hits directly into plane j
        Particle p = particles[j];

        for (int s = 0; s < STEPS_PER_TICK; s++) {
            double xp = ca * p.x + p.y + p.c;
            double cp = 2.0 * cb * xp * xp / (1.0 + xp * xp);
            double yp = ca * xp - p.x + cp;

            p.x = xp; p.y = yp; p.c = cp;

            int px = (int)(sc * xp + cx);
            int py = (int)(sc * yp + cy);

            if (px >= 0 && px < fb_w && py >= 0 && py < fb_h)
                plane[(size_t)py * fb_w + px] += PHOSPHOR_GAIN;
        }

        particles[j] = p;

        step_done.arrive_and_wait();

        // phase 2: tone-map this thread's disjoint pixel slice
        const size_t span = (NP + N_COLORS) / (N_COLORS + 1);
        const size_t lo = std::min(NP, (size_t)(j + 1) * span);
        const size_t hi = std::min(NP, lo + span);
        tone_map_range(lo, hi, NP);

        tone_done.arrive_and_wait();
    }
}

// --- stdin command pump (main thread, non-blocking) ---------------

static std::string cmd_buf;
static bool stdin_closed = false;

static void handle_line(const std::string& line)
{
    if (line.empty()) return;

    switch (line[0]) {
    case 'n': next_rose();           break;
    case 'b': back_rose();           break;
    case 'p': toggle_phosphor();     break;
    case 'f': case 'F': toggle_shading(); break;
    case '-': shorter_persistence(); break;
    case '=': longer_persistence();  break;
    case 'R': render();              break;
    case 'q': quit_requested.store(true); break;
    case 'r': {
        int w = 0, h = 0;
        if (sscanf(line.c_str() + 1, "%d %d", &w, &h) == 2 &&
            w > 0 && h > 0 && (w != fb_w || h != fb_h))
            create_accumulator(w, h);
        break;
    }
    default:
        break;
    }
}

static void pump_stdin(void)
{
    if (stdin_closed) return;

    struct pollfd pfd;
    pfd.fd = 0;
    pfd.events = POLLIN;

    if (poll(&pfd, 1, 0) <= 0) return;

    char buf[256];
    ssize_t n = read(0, buf, sizeof buf);

    if (n == 0) {            // stdin closed: keep rendering anyway
        stdin_closed = true;
        return;
    }
    if (n < 0) {
        if (errno != EINTR) stdin_closed = true;
        return;
    }

    cmd_buf.append(buf, (size_t)n);

    size_t pos;
    while ((pos = cmd_buf.find('\n')) != std::string::npos) {
        handle_line(cmd_buf.substr(0, pos));
        cmd_buf.erase(0, pos + 1);
    }
}

// --- main ----------------------------------------------------------

int main(void)
{
    // this binary is the compute core: it writes raw PPM frames
    // to stdout, so running it straight in a terminal would dump
    // binary garbage. refuse politely instead.
    if (isatty(STDOUT_FILENO)) {
        fprintf(stderr,
            "roses: this is the compute core, not the app.\n"
            "It streams PPM frames on stdout — run the display "
            "shim instead:\n\n"
            "    python3 display.py\n\n");
        return 1;
    }

    // reader (display shim) going away must not SIGPIPE us;
    // ferror(stdout) below handles it instead
    signal(SIGPIPE, SIG_IGN);

    // first rose: index 0, saved into the buffer
    double a0 = time_random(-1.0, 1.0);
    a.store(a0);
    b.store(1.0 - a0);
    ab_buffer.push_back({ a0, 1.0 - a0 });
    index = 0;

    create_accumulator(WINDOW_W, WINDOW_H);
    render();

    // one worker thread per attractor: worker j owns particle j
    // and density plane j
    std::thread workers[N_COLORS];
    for (int j = 0; j < N_COLORS; j++)
        workers[j] = std::thread(worker, j);

    for (;;) {
        // main-only window: workers are parked at step-start, so
        // commands, resizes and reseeds touch shared state safely
        pump_stdin();
        if (quit_requested.load()) break;

        step_start.arrive_and_wait();   // release workers into phase 1
        step_done.arrive_and_wait();    // wait: all planes are final

        // main tone-maps slice 0 while workers tone-map slices 1..8
        const size_t NP = (size_t)fb_w * fb_h;
        const size_t span = (NP + N_COLORS) / (N_COLORS + 1);
        tone_map_range(0, std::min(NP, span), NP);

        tone_done.arrive_and_wait();    // all slices written

        // emit frame: descriptor line + PPM
        char title[128];
        snprintf(title, sizeof title,
                 "Rose Attractor   #%d   a=%.6f   b=%.6f",
                 index, a.load(), b.load());

        char hdr[160];
        int n = snprintf(hdr, sizeof hdr,
                         "F %d %d %s\nP6\n%d %d\n255\n",
                         fb_w, fb_h, title, fb_w, fb_h);

        fwrite(hdr, 1, (size_t)n, stdout);
        fwrite(rgb.data(), 1, NP * 3, stdout);
        fflush(stdout);

        if (ferror(stdout)) break;   // display shim went away
    }

    // wake the workers parked at step-start so they can observe
    // quit_requested and exit their loops
    quit_requested.store(true);
    step_start.arrive_and_wait();
    for (int j = 0; j < N_COLORS; j++)
        workers[j].join();

    return 0;
}
