#!/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3
import os,sys
runner='/workspace/scratch/a1b3cd86fa71/session/toolchains/landrun/landrun'
args=[arg for arg in sys.argv[1:] if arg != "--best-effort"]
os.execv(runner,[runner,*args])
