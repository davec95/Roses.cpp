#!/usr/bin/env python3
"""
display.py - tkinter display shim for the C++ roses core.

C++ has no standard GUI, so the port is split: roses.cpp does ALL
attractor math and emits PPM frames on stdout; this file launches
./roses, shows its frames on a tkinter canvas, and forwards
keypresses back to the core as one-line commands. No attractor
math lives here.

Controls (identical to roses.py):
    Space      next rose
    Backspace  previous rose
    p          toggle phosphor trails (off = solid dots)
    - / =      phosphor shorter / longer persistence
    Enter      toggle fullscreen
    Escape     leave fullscreen
    q          quit

Usage: python3 display.py   (expects ./roses next to it)
"""

import os
import queue
import subprocess
import threading
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "roses")

WINDOW_W = 1280
WINDOW_H = 700

root = tk.Tk()
root.geometry(f"{WINDOW_W}x{WINDOW_H}")

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

photo = None
img_id = None

frames = queue.Queue()
core = None


# --- commands to the core ------------------------------------------

def send(cmd):
    if core is None or core.stdin is None:
        return
    try:
        core.stdin.write((cmd + "\n").encode())
        core.stdin.flush()
    except (BrokenPipeError, OSError):
        pass


def send_resize():
    w = canvas.winfo_width() or WINDOW_W
    h = canvas.winfo_height() or WINDOW_H
    send(f"r {w} {h}")


# --- frame reader (background thread: bytes only, no Tk calls) -----

def read_exactly(n):
    buf = b""
    while len(buf) < n:
        chunk = core.stdout.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def reader():
    while True:
        line = core.stdout.readline()
        if not line:
            break
        if not line.startswith(b"F "):
            continue

        parts = line.split(b" ", 3)
        try:
            w = int(parts[1])
            h = int(parts[2])
        except (IndexError, ValueError):
            continue
        title = parts[3].decode(errors="replace").rstrip("\n") \
            if len(parts) > 3 else ""

        # PPM header: magic, dims, maxval
        header = (core.stdout.readline()
                  + core.stdout.readline()
                  + core.stdout.readline())
        data = read_exactly(w * h * 3)
        if data is None:
            break

        frames.put((title, header + data))

    frames.put(None)  # sentinel: core exited


# --- frame pump (main thread: all Tk work happens here) ------------

def pump():
    global photo, img_id

    frame = None
    while True:
        try:
            f = frames.get_nowait()
        except queue.Empty:
            break
        if f is None:
            root.destroy()
            return
        frame = f          # keep only the newest frame

    if frame is not None:
        title, ppm = frame
        root.title(title)
        photo = tk.PhotoImage(data=ppm)
        if img_id is None:
            img_id = canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            canvas.itemconfig(img_id, image=photo)

    root.after(1, pump)


# --- key handlers: translate to core commands ----------------------

def next_rose(event=None):
    send_resize()
    send("n")


def back_rose(event=None):
    send_resize()
    send("b")


def toggle_phosphor(event=None):
    send("p")


def shorter_persistence(event=None):
    send("-")


def longer_persistence(event=None):
    send("=")


def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", True)
    root.update_idletasks()
    send_resize()
    send("R")


def escape_fullscreen(event=None):
    root.attributes("-fullscreen", False)
    root.geometry(f"{WINDOW_W}x{WINDOW_H}")
    root.update_idletasks()
    send_resize()
    send("R")


def quit_program(event=None):
    send("q")
    root.destroy()


root.bind("<space>", next_rose)
root.bind("<BackSpace>", back_rose)
root.bind("<Return>", toggle_fullscreen)
root.bind("<Escape>", escape_fullscreen)
root.bind("p", toggle_phosphor)
root.bind("minus", shorter_persistence)
root.bind("equal", longer_persistence)
root.bind("q", quit_program)
root.protocol("WM_DELETE_WINDOW", quit_program)


# --- launch core and run --------------------------------------------

core = subprocess.Popen(
    [CORE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

threading.Thread(target=reader, daemon=True).start()
root.after(1, pump)
root.mainloop()

if core.poll() is None:
    send("q")
    try:
        core.wait(timeout=2)
    except subprocess.TimeoutExpired:
        core.kill()
