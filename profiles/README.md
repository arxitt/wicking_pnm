These are the simulation's profiling runs, profiled with cProfile. They're all ordered by the number of calls.

They're all taken on the same machine running OpenSUSE Leap 15.1 with Conda's Python 3.8.2. They're created using the command: `python -m cProfile -s ncalls ./network_script.py -Np (args...)`. Changes to -c, -t or -j are described in the title, otherwise the program's defaults are used.

`memory_info.txt` and `cpu_info.txt` show the output of lsmem and lscpu respectively.

`archive/profileN.txt` show the profile for the default settings through previous iterations of this software. Be careful, the default timestep was changed without notice, so the earlier values should be compared with `profile_t1E-4.txt`

`profile.txt`, `profile_c12.txt` and `profile_t1E-4.txt` will be kept up to date with each meaningful change.

For now they're done with experimental data. Later versions will do it with a generated artificial pore network.
