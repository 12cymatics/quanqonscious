# QuanQonscious Simulation - Quick Start Guide

## 📁 File Locations

All files are in: `/home/user/quanqonscious/`

### Main Files:
1. **H2_GRVQ_FULL_FIXED.py** (33 KB)
   - Complete fixed simulation with MPI, CUDAq, Cirq, CUDA support
   - Production-ready code

2. **SIMULATION_FIXES_DOCUMENTATION.md** (16 KB)
   - Complete documentation of all fixes
   - Installation instructions
   - Usage examples

3. **QUANQONSCIOUS_HPC_COLAB.ipynb** (24 KB)
   - Original notebook (simplified version)
   - Missing MPI, CUDAq, CUDA features

4. **H2_MST_Dashboard_Rank3.py** (25 KB)
   - Original reference implementation

## 🚀 How to Run

### Option 1: Check the file exists
```bash
cd /home/user/quanqonscious
ls -lh H2_GRVQ_FULL_FIXED.py
```

### Option 2: View the file
```bash
cat H2_GRVQ_FULL_FIXED.py | less
# Or
nano H2_GRVQ_FULL_FIXED.py
```

### Option 3: Run the simulation (minimal setup)
```bash
pip install numpy scipy numba cirq plotly kaleido
python H2_GRVQ_FULL_FIXED.py
```

### Option 4: Run with all features (HPC)
```bash
pip install numpy scipy numba mpi4py cirq cuda-quantum plotly kaleido
mpirun -np 4 python H2_GRVQ_FULL_FIXED.py
```

## 📊 Output Files

After running, you'll get:
- **H2_GRVQ_FULL_Dashboard.html** - Interactive visualization
- **H2_GRVQ_FULL_Results.npz** - Raw data

## 🔍 Git Status

Branch: `claude/run-palindrome-simulations-011CUXRCsV1HEPmc2uArQnqo`
Commit: `63bfc43`

View on GitHub:
```bash
git log --oneline -1
git show --stat
```

## 📖 Read Documentation

```bash
cat SIMULATION_FIXES_DOCUMENTATION.md | less
```

Or open in editor:
```bash
nano SIMULATION_FIXES_DOCUMENTATION.md
```

## ✅ Verification

Check all files are present:
```bash
ls -lh | grep -E "(H2_GRVQ_FULL|SIMULATION_FIXES)"
```

Expected output:
```
-rw-r--r-- 1 root root  33K Nov  6 09:12 H2_GRVQ_FULL_FIXED.py
-rw-r--r-- 1 root root  16K Nov  6 09:14 SIMULATION_FIXES_DOCUMENTATION.md
```

## 🌐 Access via GitHub

The files are pushed to GitHub:
https://github.com/12cymatics/quanqonscious/tree/claude/run-palindrome-simulations-011CUXRCsV1HEPmc2uArQnqo

Files in this branch:
- H2_GRVQ_FULL_FIXED.py
- SIMULATION_FIXES_DOCUMENTATION.md

## 💡 What's Different from Original?

| Feature | Original Notebook | H2_GRVQ_FULL_FIXED.py |
|---------|------------------|------------------------|
| MPI | ❌ Missing | ✅ Full support |
| CUDAq | ❌ Missing | ✅ Integrated |
| CUDA GPU | ❌ Missing | ✅ Full kernels |
| Error handling | ❌ Minimal | ✅ Comprehensive |
| 5-qubit GRVQ | ❌ Missing | ✅ Complete |

## 🆘 Troubleshooting

**File not found?**
```bash
# Check current directory
pwd
# Should output: /home/user/quanqonscious

# Find the file
find . -name "H2_GRVQ_FULL_FIXED.py"
```

**Permission denied?**
```bash
chmod +x H2_GRVQ_FULL_FIXED.py
```

**Import errors?**
```bash
pip install numpy scipy numba cirq plotly kaleido
```

## 📝 Summary

✅ Files created and committed
✅ All errors fixed
✅ Production-ready code
✅ Complete documentation

Ready to run!
