# QuanQonscious/maya_cipher.py

import math
import time

class MayaCipher:
    """
    Custom Vedic quantum cipher implementing the Maya Sutra principles.
    
    This cipher uses a Feistel network with time-dependent round functions. 
    It operates on 64-bit blocks (split into two 32-bit halves) and uses sinusoidal 
    modulation in the round function for enhanced security.
    """
    def __init__(self, key: int, rounds: int = 4, use_time: bool = True):
        """
        Initialize the MayaCipher.
        
        Args:
            key: An integer key for the cipher (used to derive round keys).
            rounds: Number of Feistel rounds (default 4; can be increased for security).
            use_time: Whether to incorporate current time in the encryption (dynamic round function).
        """
        self.master_key = key
        self.rounds = rounds
        self.use_time = use_time
        # Derive round subkeys (simple derivation: use bytes of the key rotated)
        self.round_keys = self._derive_round_keys(key, rounds)
    
    def _derive_round_keys(self, key: int, rounds: int):
        """Derive a list of round keys from the master key (simple approach using key bytes)."""
        # Use 32-bit chunks of the key (if key is shorter, repeat or mix bits)
        round_keys = []
        for i in range(rounds):
            # Rotate key by 8 bits per round for variation
            k = ((key << (8 * i)) | (key >> (32 - 8 * i))) & 0xFFFFFFFF
            round_keys.append(k)
        return round_keys
    
    def _round_function(self, R: int, K: int, t: float, round_index: int) -> int:
        """
        Round function F(R, K, t) -> 32-bit output.
        Combines the right half, round key, and time-based sinusoidal modulation.
        """
        # Base combination: XOR of R and round key
        x = (R ^ K) & 0xFFFFFFFF
        # Time-based modulation terms (A*cos + B*sin). Use round_index to vary frequencies.
        if t is None:
            t = 0.0
        omega1 = 1.0 + round_index * 0.5  # frequency varies each round
        omega2 = 2.0 + round_index * 0.3
        # Compute offset with some fixed amplitudes
        A = 127; B = 127  # amplitude (127 to keep within byte range roughly)
        offset = int(A * math.cos(omega1 * t) + B * math.sin(omega2 * t)) & 0xFF
        # Combine offset into x (mod 2^32 addition)
        result = (x + offset) & 0xFFFFFFFF
        # Ensure result is 32-bit
        return result
    
    def encrypt_block(self, plaintext: int, t: float = None) -> int:
        """
        Encrypt a single 64-bit block (plaintext given as int) using the Maya cipher.
        
        Args:
            plaintext: 64-bit integer to encrypt.
            t: Optional time parameter. If None and use_time is True, current time is used.
        Returns:
            64-bit integer ciphertext.
        """
        # Split plaintext into 32-bit halves
        L = (plaintext >> 32) & 0xFFFFFFFF
        R = plaintext & 0xFFFFFFFF
        if t is None:
            t = time.time() if self.use_time else 0.0
        # Feistel rounds
        for r in range(self.rounds):
            F_out = self._round_function(R, self.round_keys[r], t, r)
            new_L = R
            new_R = L ^ F_out  # XOR operation as Feistel combine
            L, R = new_L, new_R
        # Combine halves into 64-bit
        ciphertext = ((L & 0xFFFFFFFF) << 32) | (R & 0xFFFFFFFF)
        return ciphertext
    
    def decrypt_block(self, ciphertext: int, t: float = None) -> int:
        """
        Decrypt a single 64-bit block (ciphertext given as int) using the Maya cipher.
        
        Args:
            ciphertext: 64-bit integer to decrypt.
            t: The time parameter used during encryption (must match for correct decryption).
        Returns:
            64-bit integer plaintext.
        """
        # Split ciphertext into 32-bit halves
        L = (ciphertext >> 32) & 0xFFFFFFFF
        R = ciphertext & 0xFFFFFFFF
        if t is None:
            t = time.time() if self.use_time else 0.0
        # Feistel rounds (reverse order for decryption)
        for r in reversed(range(self.rounds)):
            F_out = self._round_function(L, self.round_keys[r], t, r)
            new_R = L
            new_L = R ^ F_out
            L, R = new_L, new_R
        plaintext = ((L & 0xFFFFFFFF) << 32) | (R & 0xFFFFFFFF)
        return plaintext
    
    def encrypt_message(self, message: bytes) -> bytes:
        """
        Encrypt an arbitrary-length message (bytes) by splitting into 8-byte blocks.
        
        Args:
            message: Plaintext bytes.
        Returns:
            Ciphertext as bytes (same length as input, padded if necessary).
        """
        # Pad message to 8-byte multiple
        pad_len = (-len(message)) % 8
        message_padded = message + b'\x00' * pad_len
        ciphertext_blocks = []
        t = time.time() if self.use_time else 0.0
        for i in range(0, len(message_padded), 8):
            block = int.from_bytes(message_padded[i:i+8], byteorder='big')
            encrypted_block = self.encrypt_block(block, t=t)
            ciphertext_blocks.append(encrypted_block.to_bytes(8, byteorder='big'))
        return b''.join(ciphertext_blocks)
    
    def decrypt_message(self, ciphertext: bytes) -> bytes:
        """
        Decrypt an arbitrary-length message (bytes) that was encrypted with this cipher.
        
        Args:
            ciphertext: Ciphertext bytes (length multiple of 8).
        Returns:
            Decrypted plaintext bytes (unpadded).
        """
        assert len(ciphertext) % 8 == 0, "Ciphertext length must be multiple of 8 bytes"
        plaintext_blocks = []
        t = time.time() if self.use_time else 0.0
        for i in range(0, len(ciphertext), 8):
            block = int.from_bytes(ciphertext[i:i+8], byteorder='big')
            decrypted_block = self.decrypt_block(block, t=t)
            plaintext_blocks.append(decrypted_block.to_bytes(8, byteorder='big'))
        # Join and remove padding nulls
        plain = b''.join(plaintext_blocks)
        return plain.rstrip(b'\x00')
