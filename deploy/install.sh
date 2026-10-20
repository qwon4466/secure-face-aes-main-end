#!/usr/bin/env bash
# SecureFace-RX systemd 서비스 설치 스크립트
#
#   sudo ./deploy/install.sh                 서버 서비스 설치 후 시작
#   sudo NGROK_URL=https://<주소> ./deploy/install.sh
#                                            ngrok 터널 서비스까지 설치
#   ./deploy/install.sh --dry-run            설치하지 않고 만들어질 파일만 출력
#   sudo ./deploy/install.sh --uninstall     서비스 중지 후 제거
#
# 자동으로 정하는 값 (환경변수로 직접 지정 가능)
#   계정    SF_USER    기본: sudo를 실행한 사람
#   파이썬  SF_PYTHON  기본: 저장소 안 가상환경(hk_env, .venv, venv, myvenv, env)
#                            → 계정 홈의 ~/myvenv → 시스템 python3
#   ngrok   NGROK_BIN  기본: PATH의 ngrok
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"

die() { echo "[install] 오류: $*" >&2; exit 1; }

MODE=install
case "${1:-}" in
  "")          ;;
  --dry-run)   MODE=dry ;;
  --uninstall) MODE=uninstall ;;
  -h|--help)   sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *)           die "알 수 없는 옵션: $1 (--help 참고)" ;;
esac

if [ "$MODE" = uninstall ]; then
  [ "$(id -u)" -eq 0 ] || die "sudo로 실행하세요: sudo $0 --uninstall"
  for u in ngrok secure-face; do
    systemctl disable --now "$u.service" 2>/dev/null || true
    rm -f "$UNIT_DIR/$u.service"
  done
  systemctl daemon-reload
  echo "[install] 서비스를 제거했습니다."
  exit 0
fi

# ── 계정 ────────────────────────────────────────────────────────────────
RUN_USER="${SF_USER:-${SUDO_USER:-$(id -un)}}"
[ "$RUN_USER" != root ] || die "root 계정으로는 실행하지 않습니다. 일반 계정에서 sudo로 실행하거나 SF_USER를 지정하세요."
id "$RUN_USER" >/dev/null 2>&1 || die "존재하지 않는 계정: $RUN_USER"
RUN_GROUP="$(id -gn "$RUN_USER")"
USER_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

# ── 파이썬 ──────────────────────────────────────────────────────────────
PYTHON="${SF_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for cand in "$APP_DIR"/{hk_env,.venv,venv,myvenv,env}/bin/python "$USER_HOME/myvenv/bin/python"; do
    if [ -x "$cand" ]; then PYTHON="$cand"; break; fi
  done
fi
[ -n "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || die "파이썬을 찾지 못했습니다. SF_PYTHON=/경로/python 으로 지정하세요."

# ── 카메라·가속기 접근 그룹 (이 기기에 실제로 있는 것만) ──────────────────
SUPP_GROUPS=""
for g in video render; do
  getent group "$g" >/dev/null && SUPP_GROUPS="$SUPP_GROUPS $g"
done
SUPP_GROUPS="${SUPP_GROUPS# }"

# ── ngrok (NGROK_URL을 줬을 때만) ────────────────────────────────────────
UNITS=(secure-face)
NGROK_URL="${NGROK_URL:-}"
NGROK_BIN="${NGROK_BIN:-}"
if [ -n "$NGROK_URL" ]; then
  [ -n "$NGROK_BIN" ] || NGROK_BIN="$(command -v ngrok || true)"
  [ -n "$NGROK_BIN" ] && [ -x "$NGROK_BIN" ] || die "ngrok 실행 파일을 찾지 못했습니다. NGROK_BIN=/경로/ngrok 으로 지정하세요."
  UNITS+=(ngrok)
fi

for p in "$APP_DIR" "$PYTHON" "$NGROK_BIN"; do
  case "$p" in *[[:space:]]*) die "공백이 들어간 경로는 지원하지 않습니다: $p" ;; esac
done

# sed 치환 문자열에 들어가도 안전하게 (&, |, \ 이스케이프)
esc() { printf '%s' "$1" | sed -e 's/[&|\]/\&/g'; }

render() {
  local supp_rule
  if [ -n "$SUPP_GROUPS" ]; then
    supp_rule="s|@SUPP_GROUPS@|$(esc "$SUPP_GROUPS")|g"
  else
    supp_rule='/^SupplementaryGroups=@SUPP_GROUPS@$/d'
  fi
  sed -e '/^# /d' \
      -e "$supp_rule" \
      -e "s|@USER@|$(esc "$RUN_USER")|g" \
      -e "s|@GROUP@|$(esc "$RUN_GROUP")|g" \
      -e "s|@APP_DIR@|$(esc "$APP_DIR")|g" \
      -e "s|@PYTHON@|$(esc "$PYTHON")|g" \
      -e "s|@NGROK_BIN@|$(esc "$NGROK_BIN")|g" \
      -e "s|@NGROK_URL@|$(esc "$NGROK_URL")|g" \
      "$1"
}

echo "[install] 계정=$RUN_USER:$RUN_GROUP  그룹=${SUPP_GROUPS:-(없음)}"
echo "[install] 경로=$APP_DIR"
echo "[install] 파이썬=$PYTHON"
[ -z "$NGROK_URL" ] || echo "[install] ngrok=$NGROK_BIN → $NGROK_URL"

if [ "$MODE" = dry ]; then
  for u in "${UNITS[@]}"; do
    echo; echo "===== $UNIT_DIR/$u.service ====="
    render "$DEPLOY_DIR/$u.service.in"
  done
  exit 0
fi

[ "$(id -u)" -eq 0 ] || die "sudo로 실행하세요: sudo $0"
chmod +x "$DEPLOY_DIR/wait-for-secure-face.sh"
for u in "${UNITS[@]}"; do
  render "$DEPLOY_DIR/$u.service.in" > "$UNIT_DIR/$u.service"
  chmod 644 "$UNIT_DIR/$u.service"
  echo "[install] 설치: $UNIT_DIR/$u.service"
done
systemctl daemon-reload
systemctl enable --now "${UNITS[@]/%/.service}"
echo
echo "[install] 완료. 상태 확인:  systemctl status secure-face"
echo "[install]       로그 보기:  journalctl -u secure-face -f"
