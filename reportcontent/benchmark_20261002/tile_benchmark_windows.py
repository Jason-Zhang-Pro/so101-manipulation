"""Tile only X11 GUI windows belonging to explicitly supplied benchmark PIDs.

No application is opened or closed. Window titles are never used for matching.
The function reads EWMH client/PID properties and the current /proc process tree.
It requests 640 x 480 client areas in at most two columns. If the current desktop
work area is smaller, the client size is reduced just enough to keep all selected
windows visible; the actual geometry and this adjustment are returned.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import math
from pathlib import Path
from time import sleep
from typing import Iterable


_Atom = C.c_ulong
_Window = C.c_ulong
_Display = C.c_void_p


class _ClientData(C.Union):
    _fields_ = [("b", C.c_char * 20), ("s", C.c_short * 10), ("l", C.c_long * 5)]


class _ClientMessage(C.Structure):
    _fields_ = [("type", C.c_int), ("serial", C.c_ulong), ("send_event", C.c_int),
                ("display", _Display), ("window", _Window), ("message_type", _Atom),
                ("format", C.c_int), ("data", _ClientData)]


class _Event(C.Union):
    _fields_ = [("xclient", _ClientMessage), ("pad", C.c_long * 24)]


def _process_parents() -> dict[int, int]:
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
            # comm may contain spaces and parentheses; fields after its final
            # closing parenthesis begin with state, then PPID.
            parents[int(entry.name)] = int(stat[stat.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return parents


def _owner(pid: int, roots: set[int], parents: dict[int, int]) -> int | None:
    if pid not in parents:
        return None
    seen = set()
    while pid > 0 and pid not in seen:
        if pid in roots:
            return pid
        seen.add(pid)
        pid = parents.get(pid, 0)
    return None


class _X11:
    def __init__(self) -> None:
        library = ctypes.util.find_library("X11")
        if not library:
            raise RuntimeError("libX11 is unavailable.")
        self.lib = C.CDLL(library)
        signatures = {
            "XOpenDisplay": ([C.c_char_p], _Display),
            "XCloseDisplay": ([_Display], C.c_int),
            "XDefaultRootWindow": ([_Display], _Window),
            "XDefaultScreen": ([_Display], C.c_int),
            "XDisplayWidth": ([_Display, C.c_int], C.c_int),
            "XDisplayHeight": ([_Display, C.c_int], C.c_int),
            "XInternAtom": ([_Display, C.c_char_p, C.c_int], _Atom),
            "XGetWindowProperty": ([_Display, _Window, _Atom, C.c_long, C.c_long,
                C.c_int, _Atom, C.POINTER(_Atom), C.POINTER(C.c_int),
                C.POINTER(C.c_ulong), C.POINTER(C.c_ulong),
                C.POINTER(C.POINTER(C.c_ubyte))], C.c_int),
            "XFree": ([C.c_void_p], C.c_int),
            "XGetGeometry": ([_Display, _Window, C.POINTER(_Window),
                C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_uint),
                C.POINTER(C.c_uint), C.POINTER(C.c_uint), C.POINTER(C.c_uint)], C.c_int),
            "XTranslateCoordinates": ([_Display, _Window, _Window, C.c_int,
                C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int),
                C.POINTER(_Window)], C.c_int),
            "XSendEvent": ([_Display, _Window, C.c_int, C.c_long, C.POINTER(_Event)], C.c_int),
            "XMoveResizeWindow": ([_Display, _Window, C.c_int, C.c_int,
                                    C.c_uint, C.c_uint], C.c_int),
            "XSync": ([_Display, C.c_int], C.c_int),
            "XFlush": ([_Display], C.c_int),
            "XSetErrorHandler": ([C.c_void_p], C.c_void_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot open X11 DISPLAY; retain the desktop DISPLAY/XAUTHORITY environment.")
        # A GUI can disappear between _NET_CLIENT_LIST and a property read.
        # Xlib's default BadWindow handler terminates the entire caller process;
        # use return-code handling during this bounded operation instead.
        handler_type = C.CFUNCTYPE(C.c_int, _Display, C.c_void_p)
        self._error_handler = handler_type(lambda display, event: 0)
        self._previous_error_handler = self.lib.XSetErrorHandler(
            C.cast(self._error_handler, C.c_void_p))
        self.root = self.lib.XDefaultRootWindow(self.display)
        self.atoms: dict[str, int] = {}

    def close(self) -> None:
        if self.display:
            self.lib.XSync(self.display, 0)
            self.lib.XSetErrorHandler(self._previous_error_handler)
            self.lib.XCloseDisplay(self.display)
            self.display = None

    def atom(self, name: str) -> int:
        if name not in self.atoms:
            self.atoms[name] = int(self.lib.XInternAtom(self.display, name.encode(), 0))
        return self.atoms[name]

    def property(self, window: int, name: str, required_type: int = 0) -> list[int]:
        actual_type, actual_format = _Atom(), C.c_int()
        count, remaining = C.c_ulong(), C.c_ulong()
        data = C.POINTER(C.c_ubyte)()
        code = self.lib.XGetWindowProperty(self.display, window, self.atom(name), 0, 16384,
            0, required_type, C.byref(actual_type), C.byref(actual_format), C.byref(count),
            C.byref(remaining), C.byref(data))
        try:
            if code != 0 or not data or actual_format.value != 32:
                return []
            if required_type and actual_type.value != required_type:
                return []
            if remaining.value:
                raise RuntimeError(f"X11 property {name} exceeds the bounded read size.")
            # Xlib exposes format-32 properties as native unsigned longs even
            # on LP64 systems. Reading them as uint32 would interleave zeros.
            values = C.cast(data, C.POINTER(C.c_ulong))
            return [int(values[index]) for index in range(count.value)]
        finally:
            if data:
                self.lib.XFree(data)

    def geometry(self, window: int) -> dict[str, int]:
        root, child = _Window(), _Window()
        x, y = C.c_int(), C.c_int()
        width, height, border, depth = C.c_uint(), C.c_uint(), C.c_uint(), C.c_uint()
        if not self.lib.XGetGeometry(self.display, window, C.byref(root), C.byref(x),
            C.byref(y), C.byref(width), C.byref(height), C.byref(border), C.byref(depth)):
            raise RuntimeError(f"Cannot read geometry for window {window:#x}.")
        if not self.lib.XTranslateCoordinates(self.display, window, self.root, 0, 0,
                                              C.byref(x), C.byref(y), C.byref(child)):
            raise RuntimeError(f"Cannot resolve root coordinates for window {window:#x}.")
        return {"x": x.value, "y": y.value, "width": width.value, "height": height.value}

    def send(self, window: int, message: str, values: tuple[int, ...]) -> None:
        event = _Event()
        event.xclient.type = 33  # ClientMessage
        event.xclient.display = self.display
        event.xclient.window = window
        event.xclient.message_type = self.atom(message)
        event.xclient.format = 32
        for index, value in enumerate(values):
            event.xclient.data.l[index] = value
        if not self.lib.XSendEvent(self.display, self.root, 0,
                                  (1 << 20) | (1 << 19), C.byref(event)):
            raise RuntimeError(f"Window manager request {message} failed for {window:#x}.")

    def work_area(self) -> tuple[int, int, int, int]:
        desktop = self.property(self.root, "_NET_CURRENT_DESKTOP", 6)
        areas = self.property(self.root, "_NET_WORKAREA", 6)
        index = 4 * (desktop[0] if desktop else 0)
        if len(areas) >= index + 4 and areas[index + 2] and areas[index + 3]:
            return tuple(areas[index:index + 4])
        screen = self.lib.XDefaultScreen(self.display)
        return (0, 0, self.lib.XDisplayWidth(self.display, screen),
                self.lib.XDisplayHeight(self.display, screen))


def tile_windows(gui_root_pids: Iterable[int]) -> list[dict]:
    """Arrange matching benchmark GUI windows and return their real geometry.

    Matching requires the exact _NET_WM_PID to be a supplied PID or a current
    /proc descendant. Transient dialogs and explicitly non-normal windows are
    excluded. If there are more than four eligible windows, fail before moving
    any of them. A root with no mapped client simply contributes no result; the
    caller can retry once its GUI has completed startup.
    """
    roots = set()
    for pid in gui_root_pids:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("gui_root_pids must contain positive integer PIDs.")
        roots.add(pid)
    if not roots:
        return []
    parents = _process_parents()
    x11 = _X11()
    try:
        clients = x11.property(x11.root, "_NET_CLIENT_LIST", 33)
        normal_type = x11.atom("_NET_WM_WINDOW_TYPE_NORMAL")
        selected = []
        for window in clients:
            values = x11.property(window, "_NET_WM_PID", 6)
            if not values:
                continue
            pid = values[0]
            owner = _owner(pid, roots, parents)
            if owner is None:
                continue
            if x11.property(window, "WM_TRANSIENT_FOR", 33):
                continue
            types = x11.property(window, "_NET_WM_WINDOW_TYPE", 4)
            if types and normal_type not in types:
                continue
            geometry = x11.geometry(window)
            if geometry["width"] < 1 or geometry["height"] < 1:
                continue
            selected.append({"window_id": window, "window_id_hex": f"0x{window:x}",
                             "pid": pid, "gui_root_pid": owner})
        selected.sort(key=lambda row: (row["gui_root_pid"], row["window_id"]))
        if len(selected) > 4:
            raise RuntimeError(f"Found {len(selected)} eligible benchmark windows; expected at most four.")
        if not selected:
            return []

        # Remove maximization / fullscreen only for the explicitly matched
        # clients, otherwise many window managers ignore the size request.
        for row in selected:
            window = row["window_id"]
            x11.send(window, "_NET_WM_STATE", (0, x11.atom("_NET_WM_STATE_MAXIMIZED_HORZ"),
                     x11.atom("_NET_WM_STATE_MAXIMIZED_VERT"), 1, 0))
            x11.send(window, "_NET_WM_STATE", (0, x11.atom("_NET_WM_STATE_FULLSCREEN"), 0, 1, 0))
        x11.lib.XFlush(x11.display)
        sleep(0.15)

        area_x, area_y, area_width, area_height = x11.work_area()
        columns, rows = min(2, len(selected)), math.ceil(len(selected) / 2)
        gap = 6
        extents = {}
        for row in selected:
            value = x11.property(row["window_id"], "_NET_FRAME_EXTENTS", 6)
            extents[row["window_id"]] = tuple(value[:4]) if len(value) >= 4 else (0, 0, 0, 0)
        frame_width = max(left + right for left, right, top, bottom in extents.values())
        frame_height = max(top + bottom for left, right, top, bottom in extents.values())
        width = min(640, (area_width - gap * (columns + 1)) // columns - frame_width)
        height = min(480, (area_height - gap * (rows + 1)) // rows - frame_height)
        if width < 64 or height < 64:
            raise RuntimeError("Desktop work area is too small to display the benchmark windows.")
        pitch_x, pitch_y = width + frame_width + gap, height + frame_height + gap

        for index, row in enumerate(selected):
            window = row["window_id"]
            # Recheck ancestry immediately before moving a client so a stale
            # process-tree snapshot cannot intentionally broaden the match.
            values = x11.property(window, "_NET_WM_PID", 6)
            if not values or values[0] != row["pid"] or _owner(row["pid"], roots, _process_parents()) is None:
                raise RuntimeError(f"Window {window:#x} no longer belongs to the requested GUI PID tree.")
            left = area_x + gap + (index % columns) * pitch_x
            top = area_y + gap + (index // columns) * pitch_y
            row["requested_geometry"] = {"x": left, "y": top, "width": width, "height": height}
            row["requested_client_size"] = [640, 480]
            row["reduced_to_fit_work_area"] = width != 640 or height != 480
            row["work_area"] = {"x": area_x, "y": area_y, "width": area_width, "height": area_height}
            flags = 1 | (15 << 8) | (1 << 12)  # NW gravity, x/y/w/h, application source
            x11.send(window, "_NET_MOVERESIZE_WINDOW", (flags, left, top, width, height))
        x11.lib.XSync(x11.display, 0)
        sleep(0.30)
        for row in selected:
            row.update(x11.geometry(row["window_id"]))
            ext = x11.property(row["window_id"], "_NET_FRAME_EXTENTS", 6)
            left, right, top, bottom = tuple(ext[:4]) if len(ext) >= 4 else (0, 0, 0, 0)
            outer = {"x": row["x"] - left, "y": row["y"] - top,
                     "width": row["width"] + left + right,
                     "height": row["height"] + top + bottom}
            row["outer_geometry"] = outer
            row["within_work_area"] = (outer["x"] >= area_x and outer["y"] >= area_y
                and outer["x"] + outer["width"] <= area_x + area_width
                and outer["y"] + outer["height"] <= area_y + area_height)
        return selected
    finally:
        x11.close()


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gui_root_pids", nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(tile_windows(args.gui_root_pids), indent=2))
