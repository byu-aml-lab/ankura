This Python code relies on ctypes to get reasonable computation speeds.  To
compile the C code, run `make` in this directory.

If just running `make` didn't work, you may need to change the directory of
`lapacke_headers_location` in the Makefile.  The location given by default is
for a Fedora 22 system that had the lapack-headers package installed.
