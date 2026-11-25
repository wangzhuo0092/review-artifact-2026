The Crowd-BT baseline uses the Google Research ELO rater model. The C++
sources retain their Apache 2.0 notices. The CSV reader in `csv-parser/`
retains its separate license and corresponds to commit
`4965c9f320d157c15bc1f5a6243de116a4caf101` of
[AriaFallah/csv-parser](https://github.com/AriaFallah/csv-parser).

Run these commands from the artifact root with CMake 3.20+, Git, and a C++17
compiler. The first command fetches the recorded versions of Abseil, Eigen,
and Highway.

```sh
cmake -S _baselines/crowd_bt -B _baselines/crowd_bt/build -DCMAKE_BUILD_TYPE=Release
cmake --build _baselines/crowd_bt/build --target elo_main --parallel 2
```

For an offline build, pass `-DFETCHCONTENT_SOURCE_DIR_ABSL=...`,
`-DFETCHCONTENT_SOURCE_DIR_EIGEN=...`, and
`-DFETCHCONTENT_SOURCE_DIR_HIGHWAY=...` with paths to local copies at the
commits recorded in `CMakeLists.txt`. Set `REPAIR_CROWD_BT_BINARY` to
the path of an existing executable from these sources to reuse a build.
