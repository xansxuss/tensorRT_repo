## 這個模組是用來取得環境變數設定，用以修改主程式參數值

1. 編譯 : 
   1. shared library

      ```bash
      cmake -B build -SETENV_BUILD_SHARED=ON -DCMAKE_INSTALL_PREFIX=/opt/setenv #指定編譯動態函式庫與安裝路徑
      cmake --build build
      cmake --install build
      ```
   2. static library

      ```bash
      cmake -B build-static -SETENV_BUILD_SHARED=OFF #指定編譯靜態函式庫與安裝路徑
      cmake --build build-static
      cmake --install build-static
      ```