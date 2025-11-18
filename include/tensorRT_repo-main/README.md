# tensorRT inference Engine on GPU pipeline

## install library
1. spdlog 
   1. github: [spdlog](https://github.com/gabime/spdlog.git)
   2. install: sudo apt install libspdlog-dev
   3. complie:
      1. git clone https://github.com/gabime/spdlog.git && cd spdlog
      2. mkdir build && cd build
      3. cmake .. -DSPDLOG_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release
      4. make -j$(nproc) && suao make install
2. yaml-cpp
   1. github: [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
   2. install: sudo apt install libyaml-cpp-dev
   3. complie:
      1. git clone https://github.com/jbeder/yaml-cpp.git && cd yamlcpp
      2. mkdir build && cd build
      3. cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
      4. make -j$(nproc) && suao make install
3. CLI11 
   1. github: [CLI](https://github.com/CLIUtils/CLI11.git)
   2. complie:
      1. git clone https://github.com/CLIUtils/CLI11.git
      2. git checkout v2.3.2
      3. cmake -B build -DCLI11_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
      4. cmake --build build && sudo cmake --install build
