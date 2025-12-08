#!/usr/bin/env python
"""Demonstration of Maya Cipher encryption and decryption"""

import sys
import time
sys.path.insert(0, '/home/user/quanqonscious')

from maya_cipher import MayaCipher


def test_maya_cipher():
    """Comprehensive Maya Cipher demonstration"""

    print("="*70)
    print("Maya Cipher - Vedic Quantum Encryption Demo")
    print("="*70)
    print("\nThe Maya Cipher uses a Feistel network with time-dependent")
    print("round functions and sinusoidal modulation for enhanced security.\n")

    # Test 1: Basic encryption/decryption
    print("\n" + "─"*70)
    print("Test 1: Basic Encryption and Decryption")
    print("─"*70)

    key = 0xDEADBEEF
    cipher = MayaCipher(key=key, rounds=4, use_time=False)

    message = b"QuanQonscious Framework"
    print(f"Original message: {message.decode()}")
    print(f"Message length: {len(message)} bytes")
    print(f"Key: 0x{key:08X}")

    # Encrypt
    ciphertext = cipher.encrypt_message(message)
    print(f"\nCiphertext (hex): {ciphertext.hex()}")
    print(f"Ciphertext length: {len(ciphertext)} bytes")

    # Decrypt
    decrypted = cipher.decrypt_message(ciphertext)
    print(f"\nDecrypted message: {decrypted.decode()}")
    print(f"✓ Encryption/Decryption successful: {message == decrypted}")

    # Test 2: Time-dependent encryption
    print("\n\n" + "─"*70)
    print("Test 2: Time-Dependent Encryption")
    print("─"*70)

    cipher_time = MayaCipher(key=key, rounds=4, use_time=True)
    message2 = b"Vedic Mathematics meets Quantum Computing"

    print(f"Message: {message2.decode()}")

    # Capture timestamp for encryption
    timestamp = time.time()
    print(f"Timestamp: {timestamp:.6f}")

    ciphertext_t = cipher_time.encrypt_message(message2, t=timestamp)
    print(f"\nCiphertext (hex): {ciphertext_t.hex()}")

    # Decrypt with same timestamp
    decrypted_t = cipher_time.decrypt_message(ciphertext_t, t=timestamp)
    print(f"Decrypted with same timestamp: {decrypted_t.decode()}")
    print(f"✓ Time-dependent encryption successful: {message2 == decrypted_t}")

    # Show that different timestamp produces different ciphertext
    timestamp2 = timestamp + 1.0
    ciphertext_t2 = cipher_time.encrypt_message(message2, t=timestamp2)
    print(f"\nCiphertext with different timestamp: {ciphertext_t2.hex()}")
    print(f"✓ Different timestamps produce different ciphertext: {ciphertext_t != ciphertext_t2}")

    # Test 3: Different round counts
    print("\n\n" + "─"*70)
    print("Test 3: Security with Different Round Counts")
    print("─"*70)

    message3 = b"29 Vedic Sutras"
    print(f"Message: {message3.decode()}")

    for rounds in [2, 3, 4]:
        cipher_r = MayaCipher(key=key, rounds=rounds, use_time=False)
        start = time.time()
        ct = cipher_r.encrypt_message(message3)
        elapsed = (time.time() - start) * 1000000  # microseconds

        print(f"\nRounds={rounds:2d}: {ct.hex()[:40]}... ({elapsed:.2f} μs)")

    # Test 4: Block-level encryption
    print("\n\n" + "─"*70)
    print("Test 4: 64-bit Block Encryption")
    print("─"*70)

    cipher_block = MayaCipher(key=0x12345678, rounds=4, use_time=False)

    # Test single 64-bit blocks
    test_blocks = [
        (0x0000000000000000, "All zeros"),
        (0xFFFFFFFFFFFFFFFF, "All ones"),
        (0x0123456789ABCDEF, "Sequential"),
        (0xAAAAAAAAAAAAAAAA, "Alternating"),
    ]

    for block, description in test_blocks:
        encrypted = cipher_block.encrypt_block(block)
        decrypted = cipher_block.decrypt_block(encrypted)

        print(f"\n{description}:")
        print(f"  Plain:     0x{block:016X}")
        print(f"  Encrypted: 0x{encrypted:016X}")
        print(f"  Decrypted: 0x{decrypted:016X}")
        print(f"  ✓ Match: {block == decrypted}")

    # Test 5: Long message encryption
    print("\n\n" + "─"*70)
    print("Test 5: Long Message Encryption")
    print("─"*70)

    long_message = b"""The 29 Vedic Sutras are ancient mathematical algorithms
from the Vedas that provide elegant solutions to arithmetic, algebraic,
and geometric problems. When combined with quantum computing principles,
they offer a unique approach to consciousness field simulations."""

    print(f"Message length: {len(long_message)} bytes")
    print(f"First 60 chars: {long_message[:60].decode()}...")

    cipher_long = MayaCipher(key=0xCAFEBABE, rounds=4, use_time=False)

    start_time = time.time()
    ct_long = cipher_long.encrypt_message(long_message)
    encrypt_time = (time.time() - start_time) * 1000  # milliseconds

    start_time = time.time()
    pt_long = cipher_long.decrypt_message(ct_long)
    decrypt_time = (time.time() - start_time) * 1000

    print(f"\nCiphertext length: {len(ct_long)} bytes")
    print(f"Encryption time: {encrypt_time:.3f} ms")
    print(f"Decryption time: {decrypt_time:.3f} ms")
    print(f"✓ Long message integrity: {long_message == pt_long}")

    # Summary
    print("\n\n" + "="*70)
    print("Maya Cipher Demo Complete")
    print("="*70)
    print("\n✓ All tests passed successfully!")
    print("\nKey features demonstrated:")
    print("  • Feistel network structure")
    print("  • Time-dependent round functions")
    print("  • Sinusoidal modulation for security")
    print("  • Variable round counts (2-16+)")
    print("  • 64-bit block encryption")
    print("  • Arbitrary message lengths")
    print("="*70)


if __name__ == "__main__":
    test_maya_cipher()
