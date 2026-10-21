#!/bin/bash
# 부팅 시 시스템 시계 맞추기 (/usr/local/sbin/srx-set-clock 로 설치).
#
# 라즈베리파이 5에는 RTC 배터리가 기본으로 없어 부팅할 때마다 시계가 과거로 돌아가고,
# 학교 등 NTP(UDP 123)가 막힌 네트워크에서는 자동 동기화도 되지 않는다.
# 그래서 웹 서버 응답의 Date 헤더로 시계를 맞춘다. 시계가 틀리면 https 인증서 검사가
# 실패하므로 http를 쓴다.
#
# http 응답은 위조될 수 있고, 시계가 크게 앞으로 가면 오래된 녹화 자동 삭제가
# 녹화를 지울 수 있다. 그래서 서로 다른 두 사이트의 시각이 MAX_SKEW초 안에서 일치할
# 때만 시계를 바꾼다.
#
# SRX_CLOCK_DRY_RUN=1 이면 시계를 바꾸지 않고 결과만 출력한다.
set -u

URLS="http://www.google.com http://www.naver.com http://www.cloudflare.com http://www.daum.net"
ATTEMPTS=6
MAX_SKEW=120
MIN_EPOCH=1767225600   # 2026-01-01 00:00 UTC. 이보다 이른 응답은 잘못된 값으로 본다.

fetch_epoch() {
    local stamp
    stamp=$(curl -sI --max-time 3 "$1" | tr -d '\r' | sed -n 's/^[Dd]ate: //p' | head -n 1)
    [ -n "$stamp" ] || return 1
    date -d "$stamp" +%s 2>/dev/null
}

for attempt in $(seq 1 "$ATTEMPTS"); do
    if [ "$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" = "yes" ]; then
        echo "[CLOCK] NTP로 이미 동기화됨: $(date '+%F %T %Z')"
        exit 0
    fi
    first=""
    first_url=""
    for url in $URLS; do
        epoch=$(fetch_epoch "$url") || continue
        [ "$epoch" -ge "$MIN_EPOCH" ] || continue
        if [ -z "$first" ]; then
            first=$epoch
            first_url=$url
            continue
        fi
        skew=$(( epoch > first ? epoch - first : first - epoch ))
        if [ "$skew" -le "$MAX_SKEW" ]; then
            if [ "${SRX_CLOCK_DRY_RUN:-0}" = "1" ]; then
                echo "[CLOCK][DRY-RUN] $first_url / $url 일치(차이 ${skew}초): $(date -d "@$epoch" '+%F %T %Z')"
                exit 0
            fi
            if date -s "@$epoch" >/dev/null; then
                echo "[CLOCK] $first_url / $url 기준으로 시계 설정(차이 ${skew}초): $(date '+%F %T %Z')"
                exit 0
            fi
        else
            echo "[CLOCK][WARN] $first_url 과 $url 시각이 ${skew}초 차이 나서 무시합니다"
        fi
    done
    sleep 5
done

echo "[CLOCK][WARN] 시계를 맞추지 못했습니다 (네트워크 확인 필요). 현재: $(date '+%F %T %Z')"
exit 0   # 실패해도 카메라 서버 시작은 막지 않는다
