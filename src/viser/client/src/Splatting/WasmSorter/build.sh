#!/usr/bin/env bash

emcc --bind -Oz sorter.cpp -o Sorter.mjs -s WASM=1 -s NO_EXIT_RUNTIME=1 -s "EXPORTED_RUNTIME_METHODS=['addOnPostRun']" -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=1GB -msimd128;
