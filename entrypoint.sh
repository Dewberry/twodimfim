# --- Accept JSON payload as argument or stdin ---
if [[ $# -gt 0 ]]; then
  PAYLOAD="$1"
  printf '%s' "$PAYLOAD" | ./payload-writer 2>&1 || true
else
  PAYLOAD="$(cat)"
  printf '%s' "$PAYLOAD" | ./payload-writer 2>&1 || true
fi

# add your call or script here, for example