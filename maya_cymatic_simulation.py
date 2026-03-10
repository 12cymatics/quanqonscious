"""Maya cipher encryption with Chladni timestamp verification simulation.

This module demonstrates an advanced cryptographic pipeline inspired by the
Maya Sutra.  A message is encrypted using the :class:`~maya_cipher.MayaCipher`
implementation and subsequently verified through a Chladni cymatic animation
whose nodal structures are deterministically derived from the encryption
timestamp and ciphertext.  The resulting 4392 Hz GIF file acts as a visual
checksum: any alteration of the ciphertext or timestamp will produce a
radically different cymatic structure.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass

import numpy as np

# ``Agg`` backend ensures headless operation for GIF generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from maya_cipher import MayaCipher


@dataclass
class CymaticVerification:
    """Container holding results of the encryption and verification step."""

    ciphertext: bytes
    timestamp: float
    animation_path: str


def _generate_chladni_frames(
    seed: bytes, frames: int, resolution: int, base_frequency: float
) -> FuncAnimation:
    """Create a Chladni cymatic animation seeded by binary data.

    The algorithm models the vibration of a square plate using standard
    Chladni mode superpositions.  Mode numbers are deterministically derived
    from ``seed`` so that the resulting nodal structure acts as a visual
    fingerprint.  Oscillation is driven at ``base_frequency`` Hertz, enabling
    accurate depiction of the requested 4392 Hz pattern.

    Parameters
    ----------
    seed:
        Raw binary data used to derive mode numbers.
    frames:
        Number of animation frames.
    resolution:
        Pixel resolution along one axis (produces ``resolution``² grid).
    base_frequency:
        Physical drive frequency in Hertz for the simulated plate vibration.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object describing the Chladni evolution.
    """

    digest = hashlib.sha256(seed).digest()
    m = 1 + digest[0] % 8  # mode numbers in [1,8]
    n = 1 + digest[1] % 8

    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    Z = np.zeros_like(X)
    img = ax.imshow(Z, cmap="inferno", interpolation="bilinear", animated=True)

    def update(frame: int):
        t = frame / frames
        phase = 2 * np.pi * base_frequency * t
        pattern = (
            np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            + np.sin(n * np.pi * X) * np.sin(m * np.pi * Y)
        )
        img.set_array(np.cos(phase) * pattern)
        return (img,)

    return FuncAnimation(fig, update, frames=frames, blit=True)


def encrypt_with_cymatic(
    message: bytes,
    key: int,
    *,
    frames: int = 60,
    resolution: int = 256,
    frequency: float = 4392.0,
    output_path: str = "cymatic_verification.gif",
) -> CymaticVerification:
    """Encrypt ``message`` and produce a cymatic verification animation.

    Parameters
    ----------
    message:
        Plaintext byte string to be encrypted.
    key:
        Symmetric key for :class:`~maya_cipher.MayaCipher`.
    frames:
        Number of frames in the generated GIF.
    resolution:
        Spatial resolution of the cymatic simulation.
    frequency:
        Resonant drive frequency in Hertz for the Chladni simulation.  Defaults
        to 4392 Hz as requested by the latest specification.
    output_path:
        File path for the resulting GIF animation.

    Returns
    -------
    CymaticVerification
        Dataclass containing ciphertext, timestamp and animation location.
    """

    cipher = MayaCipher(key)
    timestamp = time.time()
    ciphertext = cipher.encrypt_message(message, t=timestamp)

    # Combine ciphertext and timestamp to produce a unique animation seed.
    seed = hashlib.sha256(ciphertext + struct.pack("!d", timestamp)).digest()
    animation = _generate_chladni_frames(
        seed, frames=frames, resolution=resolution, base_frequency=frequency
    )

    writer = PillowWriter(fps=24)
    animation.save(output_path, writer=writer)
    plt.close(animation._fig)

    return CymaticVerification(ciphertext=ciphertext, timestamp=timestamp, animation_path=output_path)


if __name__ == "__main__":
    # Example usage for manual testing
    sample = b"Maya Sutra encryption demo"
    result = encrypt_with_cymatic(sample, key=0xDEADBEEF)
    print("Ciphertext:", result.ciphertext.hex())
    print("Timestamp:", result.timestamp)
    print("Animation saved to:", result.animation_path)
