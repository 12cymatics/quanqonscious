#!/usr/bin/env python3
"""
audio_output.py

Standalone audio synth for QuanQonscious implementing an FM-style HyperCube engine.
- Listens for JSON UDP messages on localhost:50007 to receive operator/base/mod matrices/mix mode.
- Implements operators, modulation matrix, input matrix, and three mix modes: concurrent, parallel, serial.
- Uses sounddevice for low-latency output; falls back to simpleaudio when unavailable.
- Safety clamps frequencies and levels and uses soft clipping.
- Message format examples (JSON):
  {"cmd":"update","base_ops":[110,220,...],"levels":[0.5,...],"mod_matrix": [[...]],"input_matrix":[[...]],"mix_mode":"concurrent"}
  {"cmd":"start"}, {"cmd":"stop"}, {"cmd":"quit"}
"""

import sys
import math
import json
import threading
import socket
import time
from typing import List

SAMPLE_RATE = 48000
BLOCKSIZE = 256
UDP_PORT = 50007
UDP_ADDR = ('127.0.0.1', UDP_PORT)

# Backend detection
try:
    import sounddevice as sd
    import numpy as np
    BACKEND = 'sounddevice'
except Exception:
    BACKEND = 'simpleaudio'
    import numpy as np
    try:
        import simpleaudio as sa
    except Exception:
        sa = None

# --- FM-style engine -------------------------------------------------

def clamp(v, a, b):
    return max(a, min(b, v))

class Operator:
    def __init__(self, base_freq=110.0, level=0.5):
        self.base_freq = float(base_freq)
        self.level = float(level)
        self.phase = 0.0
        self.prev_out = 0.0

class HyperCubeAudio:
    def __init__(self, num_ops=8):
        self.num_ops = max(1, int(num_ops))
        self.ops: List[Operator] = [Operator(110.0 + i*55.0, 0.2) for i in range(self.num_ops)]
        # modulation matrix target_i <- source_j
        self.mod_matrix = [[0.0 for _ in range(self.num_ops)] for _ in range(self.num_ops)]
        # input matrix flexible
        self.input_matrix = [[0.0] for _ in range(self.num_ops)]
        self.mix_mode = 'concurrent'
        self.running = False
        self.lock = threading.Lock()
        self.master_gain = 0.25
        # internal buffers
        self.last_out = [0.0]*self.num_ops

    def set_base_ops(self, bases: List[float]):
        with self.lock:
            for i, f in enumerate(bases[:self.num_ops]):
                self.ops[i].base_freq = float(f)

    def set_levels(self, levels: List[float]):
        with self.lock:
            for i, l in enumerate(levels[:self.num_ops]):
                self.ops[i].level = float(clamp(l, 0.0, 1.0))

    def set_mod_matrix(self, mat):
        with self.lock:
            if isinstance(mat, list) and len(mat) == self.num_ops and all(len(r)==self.num_ops for r in mat):
                self.mod_matrix = [[float(x) for x in row] for row in mat]

    def set_input_matrix(self, mat):
        with self.lock:
            # accept lists of length num_ops or any rectangular
            self.input_matrix = [[float(x) for x in row] if isinstance(row, list) else [float(row)] for row in mat]

    def set_mix_mode(self, mode: str):
        if mode in ('concurrent','parallel','serial'):
            with self.lock:
                self.mix_mode = mode

    def compute_frame(self, n_frames: int):
        # produce mono block n_frames length as numpy array
        out = np.zeros(n_frames, dtype=np.float32)
        dt = 1.0 / SAMPLE_RATE
        with self.lock:
            ops_copy = [Operator(op.base_freq, op.level) for op in self.ops]
            mod = [row[:] for row in self.mod_matrix]
            inp = [row[:] for row in self.input_matrix]
            mix = self.mix_mode

        # initial outputs used as modulators
        src_outs = [np.zeros(n_frames, dtype=np.float32) for _ in range(self.num_ops)]

        # Two-pass approach for concurrent: compute base sine for all then apply modulation
        # For serial/parallel we incrementally build chain
        if mix == 'concurrent':
            # first pass: base
            for i in range(self.num_ops):
                f = clamp(ops_copy[i].base_freq, 20.0, 18000.0)
                phase = ops_copy[i].phase
                for t in range(n_frames):
                    src_outs[i][t] = math.sin(phase) * ops_copy[i].level
                    phase += 2.0*math.pi*f*dt
                ops_copy[i].phase = phase % (2.0*math.pi)
            # compute frequency modulation from matrix
            freq_mods = [np.zeros(n_frames, dtype=np.float32) for _ in range(self.num_ops)]
            for i in range(self.num_ops):
                for j in range(self.num_ops):
                    amt = mod[i][j]
                    if amt == 0.0: continue
                    freq_mods[i] += amt * src_outs[j]
                # input matrix adds bias if present
                if i < len(inp):
                    bias = sum(inp[i]) / len(inp[i])
                    freq_mods[i] += bias
            # re-render with FM
            for i in range(self.num_ops):
                fbase = ops_copy[i].base_freq
                phase = ops_copy[i].phase
                level = ops_copy[i].level
                for t in range(n_frames):
                    f = clamp(fbase * (1.0 + 0.5*float(freq_mods[i][t])), 20.0, 18000.0)
                    out[t] += math.sin(phase) * level
                    phase += 2.0*math.pi*f*dt
                ops_copy[i].phase = phase % (2.0*math.pi)

        elif mix == 'parallel':
            # each operator gets modulation from lower index ops
            for i in range(self.num_ops):
                phase = ops_copy[i].phase
                fbase = ops_copy[i].base_freq
                level = ops_copy[i].level
                for t in range(n_frames):
                    # sum mod from 0..i
                    fm = 0.0
                    for j in range(0,i+1):
                        fm += mod[i][j] * (src_outs[j][t] if src_outs[j].size else 0.0)
                    # input bias
                    if i < len(inp):
                        fm += (sum(inp[i]) / len(inp[i]))
                    f = clamp(fbase * (1.0 + 0.45*fm), 20.0, 18000.0)
                    val = math.sin(phase) * level
                    out[t] += val
                    # update immediate src_outs for later operators
                    src_outs[i][t] = val
                    phase += 2.0*math.pi*f*dt
                ops_copy[i].phase = phase % (2.0*math.pi)

        else: # serial
            # chain operators: output of op i modulates op i+1
            prev = np.zeros(n_frames, dtype=np.float32)
            for i in range(self.num_ops):
                phase = ops_copy[i].phase
                fbase = ops_copy[i].base_freq
                level = ops_copy[i].level
                for t in range(n_frames):
                    fm = prev[t] * mod[i][i] if i < len(mod) and i < len(mod[i]) else 0.0
                    if i < len(inp): fm += (sum(inp[i]) / len(inp[i]))
                    f = clamp(fbase * (1.0 + 0.5*fm), 20.0, 18000.0)
                    val = math.sin(phase) * level
                    out[t] += val
                    phase += 2.0*math.pi*f*dt
                    prev[t] = val
                ops_copy[i].phase = phase % (2.0*math.pi)

        # apply master gain and soft clip
        out = np.tanh(out * self.master_gain)
        return out

    # UDP listener to receive control messages
    def udp_listener(self, host='127.0.0.1', port=UDP_PORT):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(0.5)
        print(f"[audio_output] Listening UDP {host}:{port}")
        while True:
            try:
                data, addr = sock.recvfrom(65536)
                txt = data.decode('utf8')
                msg = json.loads(txt)
                cmd = msg.get('cmd')
                if cmd == 'quit':
                    print('[audio_output] quit received')
                    self.running = False
                    break
                if cmd == 'start':
                    self.running = True
                if cmd == 'stop':
                    self.running = False
                if cmd == 'update':
                    if 'base_ops' in msg:
                        self.set_base_ops(msg['base_ops'])
                    if 'levels' in msg:
                        self.set_levels(msg['levels'])
                    if 'mod_matrix' in msg:
                        self.set_mod_matrix(msg['mod_matrix'])
                    if 'input_matrix' in msg:
                        self.set_input_matrix(msg['input_matrix'])
                    if 'mix_mode' in msg:
                        self.set_mix_mode(msg['mix_mode'])
                # respond optionally
            except socket.timeout:
                continue
            except Exception as e:
                print('[audio_output] UDP listener error', e)

    # audio loop using sounddevice or simpleaudio
    def run(self):
        self.running = True
        # start UDP listener thread
        t = threading.Thread(target=self.udp_listener, daemon=True)
        t.start()
        if BACKEND == 'sounddevice':
            import numpy as np
            def callback(outdata, frames, time_info, status):
                block = self.compute_frame(frames)
                outdata[:] = block.reshape(-1,1)
            with sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype='float32', callback=callback):
                while self.running:
                    time.sleep(0.1)
        else:
            # simpleaudio fallback
            import numpy as np
            while self.running:
                block = self.compute_frame(BLOCKSIZE)
                pcm = (block * 32767.0).astype('int16')
                if sa is not None:
                    try:
                        sa.play_buffer(pcm.tobytes(), 1, 2, SAMPLE_RATE).wait_done()
                    except Exception:
                        time.sleep(BLOCKSIZE / SAMPLE_RATE)
                else:
                    time.sleep(BLOCKSIZE / SAMPLE_RATE)

# Entrypoint
if __name__ == '__main__':
    hc = HyperCubeAudio(num_ops=12)
    try:
        hc.run()
    except KeyboardInterrupt:
        print('audio_output: interrupted')
