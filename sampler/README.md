This Python code relies on ctypes to get reasonable computation speeds.  The
lapack header files are in this directory since gcc can't seem to find the
headers from the package manager (lapack-header on Fedora).

To compile the C code, run `make` in this directory.
