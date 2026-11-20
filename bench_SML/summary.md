# Outputs: none vs full snapshots vs z-slices

Medians over 3 repetition(s). Snapshot format: bin. Post-processing timed on files just written (warm OS cache, passes >= 2).

| quantity | S (50) | M (100) | L (200) |
|---|---|---|---|
| nodes / outputs per run | 1.03e+06 / 15 | 8.12e+06 / 15 | 6.45e+07 / 15 |
| DISK bytes per output: snapshot / slice | 16.5 MB / 40.8 kB | 130 MB / 162 kB | 1.03 GB / 643 kB |
| DISK ratio snapshot / slice | 404 | 804 | 1.6e+03 |
| I/O write per output: snapshot / slice (extract + write) | 184 ms / 0.0922 ms + 2.84 ms | 1.34 s / 0.354 ms + 6.72 ms | 10.4 s / 0.181 ms + 12 ms |
| I/O write-time ratio snapshot / slice | 62.8 | 189 | 858 |
| I/O write time spread (max-min)/median: snapshot / slice | 1.1 % / 22 % | 1.7 % / 49 % | 0.056 % / 21 % |
| COMPUTE step, no output run | 0.217 ms | 1.94 ms | 14.8 ms |
| COMPUTE step after an output: snap / slice | 2.75 ms / 0.276 ms | 5.69 ms / 2.07 ms | 38.3 ms / 15.1 ms |
| COMPUTE other steps: snap / slice | 0.268 ms / 0.219 ms | 1.95 ms / 1.93 ms | 15 ms / 14.9 ms |
| time loop: none / snap / slice | 339 ms / 3.24 s / 387 ms | 2.98 s / 23.2 s / 3.11 s | 22.3 s / 179 s / 22.6 s |
| time loop spread (max-min)/median: none / snap / slice | 2.7 % / 1.2 % / 3.2 % | 0.3 % / 1.4 % / 1.7 % | 0.093 % / 0.05 % / 0.41 % |
| output overhead on the loop: snap / slice | 2.9 s / 47.7 ms | 20.2 s / 134 ms | 157 s / 226 ms |
| SIMULATION gain of slices | 2.85 s (88 %) | 20 s (87 %) | 157 s (87 %) |
| POST per file read / cut / stats: volume | 2.03 ms / 15.4 ms / 3.29 ms | 18.9 ms / 119 ms / 35.5 ms | 144 ms / 930 ms / 296 ms |
| POST per file read / stats: slice | 0.159 ms / 0.0385 ms | 0.168 ms / 0.0542 ms | 0.202 ms / 0.172 ms |
| POST all files: volume / slice | 310 ms / 2.98 ms | 2.6 s / 3.27 ms | 20.6 s / 5.63 ms |
| WORKFLOW loop + post: snap / slice | 3.55 s / 390 ms | 25.8 s / 3.11 s | 200 s / 22.6 s |
| WORKFLOW gain of slices | 3.16 s | 22.6 s | 177 s |
