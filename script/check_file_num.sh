#!/bin/bash

while true; do
    jpg_count=$(find ../bin -maxdepth 1 -type f -name '*.jpg' | wc -l)

    if [ "$jpg_count" -gt 500 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') 檔案數量 $jpg_count 超過限制，開始刪除..."

        find . -maxdepth 1 -type f -name '*.jpg' -printf '%T@ %p\n' | \
            sort -n | \
            head -n 300 | \
            cut -d' ' -f2- | \
            xargs -d '\n' rm -v
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') 檔案數量 $jpg_count，尚未超過限制。"
    fi

    sleep 10  # 每 10 秒檢查一次，可依需求調整
done