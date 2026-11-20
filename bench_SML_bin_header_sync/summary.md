# Outputs: none vs full snapshots vs z-slices

Medians over 3 repetition(s). Snapshot format: bin-header. Post-processing timed on files just written (warm OS cache, passes >= 2).

| quantity | S (50) | S (50) sync | M (100) | M (100) sync | L (200) | L (200) sync |
|---|---|---|---|---|---|---|
| nodes / outputs per run | 1.03e+06 / 15 | 1.03e+06 / 15 | 8.12e+06 / 15 | 8.12e+06 / 15 | 6.45e+07 / 15 | 6.45e+07 / 15 |
| DISK bytes per output: snapshot / slice | 4.12 MB / 40.8 kB | 4.12 MB / 40.8 kB | 32.5 MB / 162 kB | 32.5 MB / 162 kB | 258 MB / 643 kB | 258 MB / 643 kB |
| DISK ratio snapshot / slice | 101 | 101 | 201 | 201 | 401 | 401 |
| I/O write per output: snapshot / slice (extract + write) | 58.9 ms / 0.0952 ms + 3.43 ms | 59.6 ms / 0.0951 ms + 3.08 ms | 439 ms / 0.355 ms + 6.56 ms | 441 ms / 0.373 ms + 7.14 ms | 2.97 s / 0.182 ms + 11 ms | 2.97 s / 0.182 ms + 11.2 ms |
| I/O write-time ratio snapshot / slice | 16.7 | 18.8 | 63.5 | 58.6 | 266 | 261 |
| I/O write time spread (max-min)/median: snapshot / slice | 3.7 % / 30 % | 1.1 % / 24 % | 8.3 % / 21 % | 11 % / 44 % | 0.18 % / 12 % | 0.095 % / 29 % |
| COMPUTE step, no output run | 0.218 ms | 0.218 ms | 1.95 ms | 1.95 ms | 14.8 ms | 14.8 ms |
| COMPUTE step after an output: snap / slice | 2.6 ms / 0.27 ms | 2.61 ms / 0.272 ms | 13.5 ms / 2.07 ms | 13.7 ms / 2.07 ms | 37.7 ms / 15.1 ms | 37.7 ms / 15.1 ms |
| COMPUTE other steps: snap / slice | 0.236 ms / 0.22 ms | 0.239 ms / 0.219 ms | 1.94 ms / 1.93 ms | 1.94 ms / 1.93 ms | 15 ms / 14.9 ms | 15 ms / 14.9 ms |
| time loop: none / snap / slice | 341 ms / 1.31 s / 398 ms | 341 ms / 1.32 s / 391 ms | 2.99 s / 9.77 s / 3.11 s | 2.99 s / 9.81 s / 3.11 s | 22.3 s / 67.4 s / 22.5 s | 22.3 s / 67.4 s / 22.6 s |
| time loop spread (max-min)/median: none / snap / slice | 2.3 % / 2.7 % / 5.3 % | 2.3 % / 0.5 % / 4.7 % | 0.25 % / 5.9 % / 0.76 % | 0.25 % / 8 % / 1.7 % | 0.16 % / 0.11 % / 0.24 % | 0.16 % / 0.059 % / 0.46 % |
| output overhead on the loop: snap / slice | 966 ms / 57.1 ms | 982 ms / 50.3 ms | 6.78 s / 121 ms | 6.82 s / 123 ms | 45.1 s / 211 ms | 45.1 s / 223 ms |
| SIMULATION gain of slices | 909 ms (70 %) | 931 ms (70 %) | 6.66 s (68 %) | 6.7 s (68 %) | 44.9 s (67 %) | 44.9 s (67 %) |
| POST per file read / cut / stats: volume | 0.532 ms / 14.9 ms / 1.08 ms | 0.59 ms / 14.9 ms / 1.08 ms | 3.37 ms / 114 ms / 14.3 ms | 3.27 ms / 114 ms / 14.2 ms | 35.6 ms / 892 ms / 136 ms | 35.4 ms / 891 ms / 133 ms |
| POST per file read / stats: slice | 0.151 ms / 0.0256 ms | 0.149 ms / 0.0255 ms | 0.168 ms / 0.0622 ms | 0.162 ms / 0.0628 ms | 0.192 ms / 0.176 ms | 0.197 ms / 0.148 ms |
| POST all files: volume / slice | 247 ms / 2.66 ms | 248 ms / 2.66 ms | 1.98 s / 3.48 ms | 1.98 s / 3.37 ms | 15.9 s / 5.55 ms | 15.9 s / 5.16 ms |
| WORKFLOW loop + post: snap / slice | 1.55 s / 401 ms | 1.57 s / 394 ms | 11.8 s / 3.11 s | 11.8 s / 3.11 s | 83.3 s / 22.5 s | 83.3 s / 22.6 s |
| WORKFLOW gain of slices | 1.15 s | 1.18 s | 8.64 s | 8.68 s | 60.8 s | 60.8 s |
