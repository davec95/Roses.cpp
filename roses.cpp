/*
 * roses.cpp - single-threaded C++ port of Roses.py/roses.py
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
 * This file is the compute core only: pure C++17 standard
 * library, no third-party dependencies, single-threaded
 * (check-in R1). It renders frames and speaks a tiny pipe
 * protocol; display.py (tkinter) is the display shim that
 * launches this binary, shows its frames, and forwards
 * keypresses back as commands.
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
 *       -            phosphor shorter persistence
 *       =            phosphor longer persistence
 *       r <w> <h>    set framebuffer size (ignored if unchanged)
 *       R            reseed particles (render)
 *       q            quit
 *
 * ---------------------------------------------------------------
 * DENSITY LAYOUT (matters for the R2 multithreaded step)
 *
 * The Python original stores density interleaved as
 * (h, w, 8) and uses np.bincount to sum repeated hits, because
 * numpy fancy-index += applies once per unique index. Neither
 * workaround is needed here: density is stored PLANE-MAJOR,
 * plane j at offset j*w*h, and hits accumulate with a direct
 * `plane[idx] += 1.0f`. Plane-major also means thread j can
 * later own plane j with zero false sharing and zero locks.
 *
 * Build:  g++ -O2 -std=c++17 -o roses roses.cpp
 * Run:    python3 display.py        (launches ./roses itself)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <deque>
#include <random>
#include <string>
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

// --- state --------------------------------------------------------

// density is PLANE-MAJOR: plane j occupies
// density[j*w*h .. (j+1)*w*h). See header comment.
static std::vector<float>   density;
static std::vector<uint8_t> rgb;          // tone-mapped frame
static int fb_w = 0;
static int fb_h = 0;

struct Particle { double x, y, c; };
static Particle particles[N_COLORS];

static double a = 0.0;
static double b = 0.0;
static bool   phosphor = true;
static int    decay_i = 2;

static std::deque<std::pair<double,double>> ab_buffer;  // history window
static int index = 0;                                    // cursor into it

static bool quit_requested = false;

// --- random -------------------------------------------------------

/*
 * time_random: uniform(lo, hi) with the current time folded into
 * the seed, mirroring the Python original -- wall clock in
 * nanoseconds XOR 64 bits of OS entropy, so two roses drawn at
 * different instants essentially never share the same (a, b).
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

// --- accumulator / render -----------------------------------------

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
    for (int i = 0; i < N_COLORS; i++) {
        double t = 2.0 * i - 0.51;
        double x = t;
        double y = t;
        double c = 2.0 * b * x * x / (1.0 + x * x);
        particles[i] = { x, y, c };
    }
}

// --- rose history (same sliding-window semantics as roses.py) -----

static void next_rose(void)
{
    if (index < (int)ab_buffer.size() - 1) {
        // cursor not at newest: replay the buffered rose exactly
        index++;
        a = ab_buffer[index].first;
        b = ab_buffer[index].second;
    } else {
        // cursor at newest: THIS is where a new random rose is allowed
        a = time_random(-1.0, 1.0);
        b = 1.0 - a;

        ab_buffer.push_back({ a, b });
        if ((int)ab_buffer.size() > BUFFER_SIZE)
            ab_buffer.pop_front();      // window full: drop oldest

        index = (int)ab_buffer.size() - 1;
    }

    render();
}

static void back_rose(void)
{
    if (index > 0) {
        index--;
        a = ab_buffer[index].first;
        b = ab_buffer[index].second;
    }
    render();
}

static void toggle_phosphor(void)
{
    phosphor = !phosphor;
    std::fill(density.begin(), density.end(), 0.0f);  // clear trails
}

static void shorter_persistence(void)
{
    decay_i = std::max(0, decay_i - 1);
}

static void longer_persistence(void)
{
    decay_i = std::min(N_DECAY - 1, decay_i + 1);
}

// --- one tick: decay, step+accumulate, tone-map, emit frame -------

static void iterate(void)
{
    const size_t NP = (size_t)fb_w * fb_h;
    const int cx = fb_w / 2;
    const int cy = fb_h / 2;

    // phosphor decay: every plane dims a little each tick
    if (phosphor) {
        const float k = DECAY_STEPS[decay_i];
        for (size_t i = 0; i < density.size(); i++)
            density[i] *= k;
    } else {
        std::fill(density.begin(), density.end(), 0.0f);
    }

    // step all 8 particles STEPS_PER_TICK times each and accumulate
    // hits directly: plane-major layout means `plane[idx] += 1` is
    // exact for repeated hits (no numpy bincount workaround needed)
    for (int s = 0; s < STEPS_PER_TICK; s++) {
        for (int j = 0; j < N_COLORS; j++) {
            double x = particles[j].x;
            double y = particles[j].y;
            double c = particles[j].c;

            double xp = a * x + y + c;
            double cp = 2.0 * b * xp * xp / (1.0 + xp * xp);
            double yp = a * xp - x + cp;

            particles[j] = { xp, yp, cp };

            int px = (int)(SCALE * xp + cx);
            int py = (int)(SCALE * yp + cy);

            if (px >= 0 && px < fb_w && py >= 0 && py < fb_h)
                density[(size_t)j * NP + (size_t)py * fb_w + px]
                    += PHOSPHOR_GAIN;
        }
    }

    // tone-map each plane separately, composite additively.
    // empty pixels (d == 0) skip the expf entirely -- most pixels
    // are empty, so this is a large constant-factor win.
    for (size_t p = 0; p < NP; p++) {
        float r = 0.0f, g = 0.0f, bl = 0.0f;

        for (int j = 0; j < N_COLORS; j++) {
            float d = density[(size_t)j * NP + p];
            if (d <= 0.0f) continue;

            float inten = phosphor
                ? 255.0f * (1.0f - expf(-d / GLOW_K))
                : fminf(d, 1.0f) * 255.0f;

            r  += inten * COLOR_NORM[j][0];
            g  += inten * COLOR_NORM[j][1];
            bl += inten * COLOR_NORM[j][2];
        }

        rgb[3*p    ] = (uint8_t)std::min(255.0f, std::max(0.0f, r));
        rgb[3*p + 1] = (uint8_t)std::min(255.0f, std::max(0.0f, g));
        rgb[3*p + 2] = (uint8_t)std::min(255.0f, std::max(0.0f, bl));
    }

    // emit frame: descriptor line + PPM
    char title[128];
    snprintf(title, sizeof title,
             "Rose Attractor   #%d   a=%.6f   b=%.6f",
             index, a, b);

    char hdr[160];
    int n = snprintf(hdr, sizeof hdr,
                     "F %d %d %s\nP6\n%d %d\n255\n",
                     fb_w, fb_h, title, fb_w, fb_h);

    fwrite(hdr, 1, (size_t)n, stdout);
    fwrite(rgb.data(), 1, NP * 3, stdout);
    fflush(stdout);
}

// --- stdin command pump (non-blocking, single-threaded) -----------

static std::string cmd_buf;
static bool stdin_closed = false;

static void handle_line(const std::string& line)
{
    if (line.empty()) return;

    switch (line[0]) {
    case 'n': next_rose();           break;
    case 'b': back_rose();           break;
    case 'p': toggle_phosphor();     break;
    case '-': shorter_persistence(); break;
    case '=': longer_persistence();  break;
    case 'R': render();              break;
    case 'q': quit_requested = true; break;
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
    a = time_random(-1.0, 1.0);
    b = 1.0 - a;
    ab_buffer.push_back({ a, b });
    index = 0;

    create_accumulator(WINDOW_W, WINDOW_H);
    render();

    for (;;) {
        pump_stdin();
        if (quit_requested) break;

        iterate();

        if (ferror(stdout)) break;   // display shim went away
    }

    return 0;
}
