#!/bin/bash

LOG_FILE="crash_log.txt"
CRASH_COUNT=0

export CUSTOMS_LEVEL_DEBUG="error"
export IMSHOW_FLAG=TRUE

echo "開始測試 $(date)" >> "$LOG_FILE"

rm "$LOG_FILE"

while true; do
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    ./tensorrt_infer -c ../config/config.yaml
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        ((CRASH_COUNT++))
        echo "[$START_TIME] Crash #$CRASH_COUNT, Exit Code: $EXIT_CODE" >> "$LOG_FILE"
    fi

    # 可以加一點 delay 避免系統過熱或造成過多資源損耗
    sleep 0.5
done
