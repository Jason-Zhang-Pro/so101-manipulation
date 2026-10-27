"""Read only the private preview window and save its actual displayed pixels."""
from pathlib import Path
import ctypes as C
import json
import re
import subprocess
import numpy as np
from PIL import Image

folder = Path(__file__).resolve().parent
metadata = json.loads((folder / 'scene_manifest.json').read_text())
partition = metadata['partition']
pids = set()
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        env = (path / 'environ').read_bytes().split(b'\0')
        if f'IGN_PARTITION={partition}'.encode() in env:
            pids.add(int(path.name))
    except (OSError, PermissionError):
        pass
tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
window = None
for line in tree.splitlines():
    if '"Gazebo"' not in line:
        continue
    candidate = re.search(r'0x[0-9a-f]+', line).group()
    prop = subprocess.check_output(['xprop', '-id', candidate, '_NET_WM_PID'], text=True)
    match = re.search(r'= (\d+)', prop)
    if match and int(match.group(1)) in pids:
        window = int(candidate, 16)
        print('Private preview window:', line.strip())
        break
if window is None:
    raise SystemExit('Private preview window not found.')

class XImage(C.Structure):
    _fields_ = [('width', C.c_int), ('height', C.c_int), ('xoffset', C.c_int),
                ('format', C.c_int), ('data', C.c_void_p), ('byte_order', C.c_int),
                ('bitmap_unit', C.c_int), ('bitmap_bit_order', C.c_int),
                ('bitmap_pad', C.c_int), ('depth', C.c_int),
                ('bytes_per_line', C.c_int), ('bits_per_pixel', C.c_int),
                ('red_mask', C.c_ulong), ('green_mask', C.c_ulong), ('blue_mask', C.c_ulong)]
x = C.CDLL('libX11.so.6')
x.XOpenDisplay.restype = C.c_void_p
d = x.XOpenDisplay(None)
if not d:
    raise SystemExit('No X display available.')
x.XGetGeometry.argtypes = [C.c_void_p, C.c_ulong] + [C.c_void_p] * 7
root = C.c_ulong()
left, top = C.c_int(), C.c_int()
width, height, border, depth = [C.c_uint() for _ in range(4)]
if not x.XGetGeometry(d, window, *[C.byref(v) for v in (root, left, top, width, height, border, depth)]):
    raise SystemExit('Cannot read preview geometry.')
x.XGetImage.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_int, C.c_uint, C.c_uint, C.c_ulong, C.c_int]
x.XGetImage.restype = C.POINTER(XImage)
ptr = x.XGetImage(d, window, 0, 0, width.value, height.value, C.c_ulong(-1), 2)
if not ptr:
    raise SystemExit('Cannot read private preview pixels.')
info = ptr.contents
if info.bits_per_pixel != 32:
    raise SystemExit(f'Unexpected pixel format: {info.bits_per_pixel}')
pixels = np.ndarray((info.height, info.width),
    dtype='<u4' if info.byte_order == 0 else '>u4',
    buffer=C.string_at(info.data, info.bytes_per_line * info.height),
    strides=(info.bytes_per_line, 4))
channels = []
for mask in (info.red_mask, info.green_mask, info.blue_mask):
    shift = (mask & -mask).bit_length() - 1
    channels.append((((pixels & mask) >> shift) * (255.0 / (mask >> shift))).astype(np.uint8))
Image.fromarray(np.stack(channels, axis=-1)).save(folder / 'preview.png')
print('Saved actual preview pixels:', folder / 'preview.png')
print('Original training PID still exists:', Path('/proc', str(metadata['training_pid_at_precheck'])).exists())
