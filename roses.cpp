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
 * The core runs in one of two modes, chosen at startup:
 *
 *   GUI mode (stdout is a TTY and DISPLAY is set): the core opens
 *   its own X11 window and renders directly into it via MIT-SHM
 *   (XShmPutImage from a shared-memory XImage). No display shim,
 *   no pipe. Keys are translated to the same commands listed
 *   below; window resizes drive 'r', the window manager's close
 *   button drives 'q', Enter/Escape toggle fullscreen.
 *
 *   Pipe mode (stdout redirected, e.g. by display.py): unchanged
 *   R1/R2 protocol, frames out on stdout, commands in on stdin.
 *
 * On exit (either mode) the core prints its iteration rate to
 * stderr. ONE ITERATION = one frame tick = one full pass over
 * all 8 color planes (the 8 attractors run in parallel, one
 * plane each; this is the unit the single-threaded Python
 * original showed as "8 colors, then again").
 *
 *   stdout, one frame at a time (pipe mode only):
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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <deque>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/extensions/XShm.h>

#include <poll.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

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
static int rose_index = 0;                                    // cursor (main only)

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

    if (rose_index < (int)ab_buffer.size() - 1) {
        // cursor not at newest: replay the buffered rose exactly
        rose_index++;
        ca = ab_buffer[rose_index].first;
        cb = ab_buffer[rose_index].second;
    } else {
        // cursor at newest: THIS is where a new random rose is allowed
        ca = time_random(-1.0, 1.0);
        cb = 1.0 - ca;

        ab_buffer.push_back({ ca, cb });
        if ((int)ab_buffer.size() > BUFFER_SIZE)
            ab_buffer.pop_front();      // window full: drop oldest

        rose_index = (int)ab_buffer.size() - 1;
    }

    a.store(ca);
    b.store(cb);
    render();
}

static void back_rose(void)
{
    if (rose_index > 0) {
        rose_index--;
        a.store(ab_buffer[rose_index].first);
        b.store(ab_buffer[rose_index].second);
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

// --- GUI mode: native X11 window ----------------------------------

/*
 * When stdout is a TTY and DISPLAY is set, the core opens its own
 * window and blits the tone-mapped rgb buffer straight into it via
 * MIT-SHM (XShmPutImage): the X server reads the pixels from a
 * shared-memory segment, so there is no encode/decode round trip
 * and no display shim. All input arrives as X events, translated
 * to exactly the commands the pipe protocol accepts. Only the
 * main thread ever touches Xlib, and only inside the main-only
 * window between barriers, so resizes reallocating density/rgb
 * remain race-free.
 */

struct Gui {
    Display* dpy = nullptr;
    int      scr = 0;
    Window   win = 0;
    Visual*  visual = nullptr;
    int      depth = 0;
    unsigned rmask = 0, gmask = 0, bmask = 0;
    int      rshift = 0, gshift = 0, bshift = 0;

    XImage*         img = nullptr;
    XShmSegmentInfo shminfo;
    bool            shm_ok = false;
    bool            have_image = false;
    uint8_t*        pixels = nullptr;    // 4 bytes/pixel, mask-packed

    Atom wm_delete = None;
    bool fullscreen = false;
};

static Gui  gui;
static bool gui_mode = false;

static bool gui_create_image(int w, int h);

static int x_error_handler(Display*, XErrorEvent* ev)
{
    // benign races with the WM (stale geometry events, SHM BadMatch)
    // must not kill the app: log and continue
    char buf[128];
    XGetErrorText(gui.dpy, (int)ev->error_code, buf, sizeof buf);
    fprintf(stderr, "roses: X11: %s (ignored)\n", buf);
    return 0;
}

static bool gui_init(int w, int h)
{
    gui.dpy = XOpenDisplay(nullptr);
    if (!gui.dpy) {
        fprintf(stderr, "roses: cannot open display '%s'\n",
                XDisplayName(nullptr));
        return false;
    }
    XSetErrorHandler(x_error_handler);

    gui.scr = DefaultScreen(gui.dpy);

    // TrueColor masks for the rgb24 -> native pixel conversion.
    // The default Visual struct itself carries the channel masks;
    // no XGetVisualInfo round trip needed (and its nitems_return
    // must never be NULL -- passing NULL segfaults Xlib).
    gui.visual = DefaultVisual(gui.dpy, gui.scr);
    gui.depth  = DefaultDepth(gui.dpy, gui.scr);
    gui.rmask  = gui.visual->red_mask;
    gui.gmask  = gui.visual->green_mask;
    gui.bmask  = gui.visual->blue_mask;

    if (gui.depth < 24 || !gui.rmask || !gui.gmask || !gui.bmask) {
        fprintf(stderr, "roses: need a >=24-bit TrueColor visual, "
                        "got depth %d\n", gui.depth);
        return false;
    }
    // masks are contiguous runs: shift = bit position of lowest set bit
    gui.rshift = std::countr_zero(gui.rmask);
    gui.gshift = std::countr_zero(gui.gmask);
    gui.bshift = std::countr_zero(gui.bmask);

    XSetWindowAttributes attrs;
    attrs.background_pixel = BlackPixel(gui.dpy, gui.scr);
    attrs.event_mask = KeyPressMask | ExposureMask |
                       StructureNotifyMask | VisibilityChangeMask;
    gui.win = XCreateWindow(gui.dpy, RootWindow(gui.dpy, gui.scr),
        0, 0, (unsigned)w, (unsigned)h, 0,
        (unsigned)gui.depth, InputOutput, gui.visual,
        CWBackPixel | CWEventMask, &attrs);

    gui.wm_delete = XInternAtom(gui.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(gui.dpy, gui.win, &gui.wm_delete, 1);

    XStoreName(gui.dpy, gui.win, "Rose Attractor");

    if (!gui_create_image(w, h)) return false;

    XMapRaised(gui.dpy, gui.win);
    XSetInputFocus(gui.dpy, gui.win, RevertToParent, CurrentTime);
    XFlush(gui.dpy);
    return true;
}

static void gui_destroy_image(void)
{
    if (!gui.have_image) return;
    if (gui.shm_ok) {
        XShmDetach(gui.dpy, &gui.shminfo);
        XDestroyImage(gui.img);          // frees the XImage struct
        shmdt(gui.shminfo.shmaddr);
        shmctl(gui.shminfo.shmid, IPC_RMID, nullptr);
    } else {
        XDestroyImage(gui.img);          // frees its own byte buffer
    }
    gui.img = nullptr;
    gui.pixels = nullptr;
    gui.have_image = false;
}

static bool gui_create_image(int w, int h)
{
    gui_destroy_image();

    gui.shm_ok = XShmQueryExtension(gui.dpy) ? true : false;

    if (gui.shm_ok) {
        // Xlib pads ZPixmap rows to 32 bits for depth>=24: a depth-24
        // visual is 4 bytes per pixel on the wire, not 3.
        size_t bytes = (size_t)w * h * (gui.depth >= 24 ? 4 : 2);
        int shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0777);
        if (shmid < 0) {
            gui.shm_ok = false;
        } else {
            gui.shminfo.shmid = shmid;
            gui.shminfo.shmaddr =
                (char*)shmat(shmid, nullptr, 0);
            if (gui.shminfo.shmaddr == (char*)-1) {
                gui.shm_ok = false;
            } else {
                gui.shminfo.readOnly = False;
                gui.img = XShmCreateImage(gui.dpy, gui.visual,
                    (unsigned)gui.depth, ZPixmap,
                    gui.shminfo.shmaddr, &gui.shminfo,
                    (unsigned)w, (unsigned)h);
                if (!gui.img || !XShmAttach(gui.dpy, &gui.shminfo)) {
                    if (gui.img) XDestroyImage(gui.img);
                    shmdt(gui.shminfo.shmaddr);
                    gui.shm_ok = false;
                }
            }
            if (!gui.shm_ok) {
                shmctl(shmid, IPC_RMID, nullptr);
                gui.shminfo.shmaddr = nullptr;
            }
        }
    }

    if (!gui.shm_ok) {
        // fallback: private buffer + XPutImage (slower, still correct)
        gui.img = XCreateImage(gui.dpy, gui.visual,
            (unsigned)gui.depth, ZPixmap, 0, nullptr,
            (unsigned)w, (unsigned)h,
            gui.depth >= 24 ? 32 : 16, 0);
        if (!gui.img) {
            fprintf(stderr, "roses: XCreateImage failed\n");
            return false;
        }
        size_t bytes = (size_t)gui.img->bytes_per_line * (size_t)h;
        gui.img->data = (char*)malloc(bytes);
        if (!gui.img->data) {
            fprintf(stderr, "roses: out of memory for XImage\n");
            return false;
        }
    }

    gui.pixels = (uint8_t*)gui.img->data;
    gui.have_image = true;
    return true;
}

static void gui_set_fullscreen(bool on)
{
    Atom state = XInternAtom(gui.dpy, "_NET_WM_STATE", False);
    Atom fs    = XInternAtom(gui.dpy,
                             "_NET_WM_STATE_FULLSCREEN", False);
    unsigned long fs_val = (unsigned long)fs;
    XChangeProperty(gui.dpy, gui.win, state, XA_ATOM, 32,
        PropModeReplace, (const unsigned char*)&fs_val, on ? 1 : 0);
    gui.fullscreen = on;
    XFlush(gui.dpy);
}

static void gui_pump(void)
{
    while (XPending(gui.dpy)) {
        XEvent ev;
        XNextEvent(gui.dpy, &ev);

        switch (ev.type) {
        case KeyPress:
            switch (XLookupKeysym(&ev.xkey, 0)) {
            case XK_space:     next_rose();           break;
            case XK_BackSpace: back_rose();           break;
            case XK_p: case XK_P: toggle_phosphor();  break;
            case XK_f: case XK_F: toggle_shading();   break;
            case XK_minus:     shorter_persistence(); break;
            case XK_equal:     longer_persistence();  break;
            case XK_R:         render();              break;
            case XK_Return: case XK_KP_Enter:
                gui_set_fullscreen(true);             break;
            case XK_Escape:
                gui_set_fullscreen(false);            break;
            case XK_q: case XK_Q:
                quit_requested.store(true);           break;
            default: break;
            }
            break;

        case ConfigureNotify: {
            int w = ev.xconfigure.width;
            int h = ev.xconfigure.height;
            if (w > 0 && h > 0 && (w != fb_w || h != fb_h)) {
                create_accumulator(w, h);
                if (!gui_create_image(w, h))
                    quit_requested.store(true);
            }
            break;
        }

        case ClientMessage:
            if ((Atom)ev.xclient.data.l[0] == gui.wm_delete)
                quit_requested.store(true);
            break;

        // window gone under us (WM killed it, or no WM to forward
        // WM_DELETE_WINDOW): there is nothing left to draw into
        case DestroyNotify:
        case UnmapNotify:
            quit_requested.store(true);
            break;

        default:
            break;
        }
    }
}

static void gui_blit(const char* title)
{
    // rgb24 -> native pixel words, using the visual's channel masks.
    // Xlib pads ZPixmap to 32 bits/pixel for depth>=24, so follow
    // the XImage's actual bits_per_pixel, not the window depth.
    const size_t NP = (size_t)fb_w * fb_h;
    if (gui.img->bits_per_pixel >= 32) {
        uint32_t* dst = (uint32_t*)gui.pixels;
        for (size_t p = 0; p < NP; p++) {
            const uint8_t* s = &rgb[3 * p];
            dst[p] = ((uint32_t)s[0] << gui.rshift) |
                     ((uint32_t)s[1] << gui.gshift) |
                     ((uint32_t)s[2] << gui.bshift);
        }
    } else {
        uint8_t* dst = gui.pixels;
        for (size_t p = 0; p < NP; p++) {
            const uint8_t* s = &rgb[3 * p];
            dst[3*p    ] = s[0];
            dst[3*p + 1] = s[1];
            dst[3*p + 2] = s[2];
        }
    }

    if (gui.shm_ok)
        XShmPutImage(gui.dpy, gui.win, DefaultGC(gui.dpy, gui.scr),
                     gui.img, 0, 0, 0, 0,
                     (unsigned)fb_w, (unsigned)fb_h, False);
    else
        XPutImage(gui.dpy, gui.win, DefaultGC(gui.dpy, gui.scr),
                  gui.img, 0, 0, 0, 0,
                  (unsigned)fb_w, (unsigned)fb_h);

    XStoreName(gui.dpy, gui.win, title);
    XFlush(gui.dpy);
}

// --- main ----------------------------------------------------------

int main(void)
{
    // mode selection: a TTY stdout means a human at a terminal ->
    // open our own X11 window. A redirected stdout means a display
    // shim (display.py) is reading PPM frames off the pipe.
    gui_mode = isatty(STDOUT_FILENO);

    // framebuffer first: gui_init builds an XImage sized from fb_w/fb_h
    create_accumulator(WINDOW_W, WINDOW_H);

    if (gui_mode && !gui_init(WINDOW_W, WINDOW_H)) {
        if (isatty(STDOUT_FILENO)) {
            fprintf(stderr,
                "roses: could not open a GUI window.\n"
                "This binary is also the compute core: it can "
                "stream PPM frames on stdout for a display shim:\n\n"
                "    ./roses | python3 display.py --stdin\n\n");
            return 1;
        }
        gui_mode = false;   // piped but X failed: pipe mode anyway
    }

    // reader (display shim) going away must not SIGPIPE us;
    // ferror(stdout) below handles it instead
    signal(SIGPIPE, SIG_IGN);

    // first rose: rose_index 0, saved into the buffer
    double a0 = time_random(-1.0, 1.0);
    a.store(a0);
    b.store(1.0 - a0);
    ab_buffer.push_back({ a0, 1.0 - a0 });
    rose_index = 0;

    create_accumulator(WINDOW_W, WINDOW_H);
    if (gui_mode && !gui_create_image(WINDOW_W, WINDOW_H))
        return 1;
    render();

    // one worker thread per attractor: worker j owns particle j
    // and density plane j
    std::thread workers[N_COLORS];
    for (int j = 0; j < N_COLORS; j++)
        workers[j] = std::thread(worker, j);

    // iteration counter: ONE iteration = one frame tick = one full
    // pass over all 8 color planes (the unit the slow single-threaded
    // Python original showed as "8 colors, then again")
    uint64_t iterations = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (;;) {
        // main-only window: workers are parked at step-start, so
        // commands, resizes and reseeds touch shared state safely
        if (gui_mode) gui_pump();
        else          pump_stdin();
        if (quit_requested.load()) break;

        step_start.arrive_and_wait();   // release workers into phase 1
        step_done.arrive_and_wait();    // wait: all planes are final

        // main tone-maps slice 0 while workers tone-map slices 1..8
        const size_t NP = (size_t)fb_w * fb_h;
        const size_t span = (NP + N_COLORS) / (N_COLORS + 1);
        tone_map_range(0, std::min(NP, span), NP);

        tone_done.arrive_and_wait();    // all slices written

        char title[128];
        snprintf(title, sizeof title,
                 "Rose Attractor   #%d   a=%.6f   b=%.6f",
                 rose_index, a.load(), b.load());

        if (gui_mode) {
            gui_blit(title);
        } else {
            // emit frame: descriptor line + PPM
            char hdr[160];
            int n = snprintf(hdr, sizeof hdr,
                             "F %d %d %s\nP6\n%d %d\n255\n",
                             fb_w, fb_h, title, fb_w, fb_h);

            fwrite(hdr, 1, (size_t)n, stdout);
            fwrite(rgb.data(), 1, NP * 3, stdout);
            fflush(stdout);

            if (ferror(stdout)) break;   // display shim went away
        }

        iterations++;
    }

    // wake the workers parked at step-start so they can observe
    // quit_requested and exit their loops
    quit_requested.store(true);
    step_start.arrive_and_wait();
    for (int j = 0; j < N_COLORS; j++)
        workers[j].join();

    // exit report: iterations/second (stderr so pipe mode's stdout
    // protocol stays clean)
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    if (secs > 0.0)
        fprintf(stderr,
                "\nroses: %llu iterations in %.1f s = %.1f iterations/sec"
                "  (1 iteration = 1 frame = all 8 color planes)\n",
                (unsigned long long)iterations, secs,
                (double)iterations / secs);

    if (gui_mode) {
        gui_destroy_image();
        XCloseDisplay(gui.dpy);
    }

    return 0;
}
