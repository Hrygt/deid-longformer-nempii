#!/bin/bash
# clinical-deid deploy reconcile (Fix B).
#
# Reconciles /opt/clinical-deid to a PINNED commit SHA stored in S3, on boot/relaunch.
# It NEVER pulls "whatever is on main" — it only ever moves prod to the exact SHA a human
# placed in the pin. So a plain service restart with an unchanged pin is a no-op, and prod
# can never silently drift to unreviewed main. To roll prod forward, a human updates the
# pin object, then reboots or restarts clinical-deid-reconcile.service.
#
# Fail-safe: any error (S3 unreachable, empty pin, clone/assert failure) logs and leaves the
# existing on-box code untouched so clinical-deid still starts. Never blocks the app.
#
# RECONCILE_DRYRUN=1 prints the intended action and mutates nothing.
#
# NOTE: the systemd unit runs a COPY of this script (from /run), so the overlay below can
# safely overwrite /opt/clinical-deid/reconcile.sh mid-run.
set -uo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/opt/clinical-deid}"
VERSION_FILE="${DEID_VERSION_FILE:-$DEPLOY_DIR/VERSION}"
PIN_S3="${RECONCILE_PIN_S3:-s3://cpt-dnn-model-artifacts-675138611834/clinical-deid/deploy/TARGET_SHA}"
REPO_URL="${RECONCILE_REPO_URL:-https://github.com/Hrygt/clinical-deid.git}"
DRYRUN="${RECONCILE_DRYRUN:-0}"

log() { echo "[reconcile] $*"; }

# 1) Intended SHA (the pin). Prod reads it from S3; RECONCILE_PIN_VALUE overrides the source
#    (for local dry-run tests and manual one-off pins). Any read failure -> fail safe, keep code.
if [ -n "${RECONCILE_PIN_VALUE:-}" ]; then
  TARGET="$(printf '%s' "$RECONCILE_PIN_VALUE" | tr -d '[:space:]')"
else
  TARGET="$(aws s3 cp "$PIN_S3" - 2>/dev/null | tr -d '[:space:]')"
fi
if [ -z "$TARGET" ]; then
  log "could not read pin $PIN_S3 (empty/unreachable) -> leaving current code in place"
  exit 0
fi

# 2) Current deployed SHA.
CURRENT="unknown"
[ -f "$VERSION_FILE" ] && CURRENT="$(tr -d '[:space:]' < "$VERSION_FILE")"
[ -z "$CURRENT" ] && CURRENT="unknown"

# 3) Already reconciled? no-op.
if [ "$CURRENT" = "$TARGET" ]; then
  log "up to date (VERSION=$CURRENT == pin=$TARGET) -> no-op"
  exit 0
fi

# 4) Dry-run: show intent, mutate nothing.
if [ "$DRYRUN" = "1" ]; then
  log "DRYRUN: would overlay $TARGET over $CURRENT (mutating nothing)"
  exit 0
fi

# 5) Stage a clone of the repo (full clone so the pinned commit is present).
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE="/tmp/clinical-deid-reconcile-$STAMP"
rm -rf "$STAGE"
if ! git clone --quiet "$REPO_URL" "$STAGE"; then
  log "clone failed -> leaving current code in place (VERSION=$CURRENT)"
  rm -rf "$STAGE"; exit 0
fi
# Belt-and-suspenders: if the pin isn't already present, try to fetch it directly.
if ! git -C "$STAGE" cat-file -e "${TARGET}^{commit}" 2>/dev/null; then
  git -C "$STAGE" fetch --quiet origin "$TARGET" 2>/dev/null || true
fi

# 6) SHA-ASSERT BEFORE OVERLAY: the tree we are about to ship MUST equal the pin.
RESOLVED="$(git -C "$STAGE" rev-parse --verify --quiet "${TARGET}^{commit}")"
if [ "$RESOLVED" != "$TARGET" ]; then
  log "ASSERT FAILED: pin $TARGET not resolvable in clone (got '${RESOLVED:-none}') -> abort, current code untouched"
  rm -rf "$STAGE"; exit 0
fi

# 7) Overlay tracked tree ONLY (git archive holds tracked files -> untracked model/ preserved).
if git -C "$STAGE" archive --format=tar "$TARGET" | tar -x -C "$DEPLOY_DIR"; then
  echo "$TARGET" > "$VERSION_FILE"
  log "reconciled $CURRENT -> $TARGET"
else
  log "overlay failed -> VERSION left at $CURRENT for visibility"
fi
rm -rf "$STAGE"
exit 0
