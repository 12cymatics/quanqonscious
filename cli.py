# QuanQonscious/cli.py

import argparse
import numpy as np
# Import package modules. The package directory is lowercase
# but some older scripts used a capitalized name.
from quanqonscious import ansatz, maya_cipher, updater, zpe_solver

def main():
    parser = argparse.ArgumentParser(prog="quanqonscious", 
                                     description="CLI for QuanQonscious Quantum-Consciousness Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Sub-command: run a demo simulation (e.g., one step of ZPE solver)
    sim_parser = subparsers.add_parser("simulate", help="Run a sample ZPE field simulation step.")
    sim_parser.add_argument("--size", type=int, nargs=3, metavar=("NX","NY","NZ"), default=[20,20,20],
                            help="Grid size for simulation (default 20x20x20).")
    sim_parser.add_argument("--steps", type=int, default=1, help="Number of time steps to run (default 1).")
    sim_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available.")
    # Sub-command: encrypt using Maya cipher
    enc_parser = subparsers.add_parser("encrypt", help="Encrypt a message with Maya cipher.")
    enc_parser.add_argument("key", type=int, help="Encryption key (integer).")
    enc_parser.add_argument("message", type=str, help="Message to encrypt (in quotes).")
    # Sub-command: decrypt using Maya cipher
    dec_parser = subparsers.add_parser("decrypt", help="Decrypt a message with Maya cipher.")
    dec_parser.add_argument("key", type=int, help="Decryption key (integer).")
    dec_parser.add_argument("ciphertext", type=str, help="Hex string of ciphertext to decrypt.")
    # Sub-command: update dependencies
    upd_parser = subparsers.add_parser("update", help="Update QuanQonscious dependencies to latest versions.")
    upd_parser.add_argument("--package", type=str, help="Name of a specific package to update (if not provided, update all).")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        shape = tuple(args.size)
        solver = zpe_solver.ZPEFieldSolver(shape, use_gpu=args.gpu)
        # Set a simple initial condition (e.g., a small Gaussian at center)
        cx, cy, cz = (s//2 for s in shape)
        solver.set_initial_field(lambda X, Y, Z: 
                                 1.0 * np.exp(-0.1 * ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2))
                                 if 'np' in globals() else 0.0)
        solver.step(args.steps)
        field = solver.get_field()
        print(f"Simulation completed. Field sample (center value): {field[cx, cy, cz] if field is not None else 'N/A'}")
    
    elif args.command == "encrypt":
        cipher = maya_cipher.MayaCipher(key=args.key)
        ciphertext_bytes = cipher.encrypt_message(args.message.encode('utf-8'))
        # Print as hex string
        print(ciphertext_bytes.hex())
    
    elif args.command == "decrypt":
        cipher = maya_cipher.MayaCipher(key=args.key)
        try:
            ciphertext_bytes = bytes.fromhex(args.ciphertext)
        except Exception as e:
            print(f"Error: ciphertext must be a hex string. {e}")
            return
        plaintext_bytes = cipher.decrypt_message(ciphertext_bytes)
        print(plaintext_bytes.decode('utf-8', errors='ignore'))
    
    elif args.command == "update":
        if args.package:
            updater.update_package(args.package)
        else:
            updater.update_all()

if __name__ == "__main__":
    main()
