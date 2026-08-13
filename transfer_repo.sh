#!/usr/bin/env bash
#
# Mirror the repository to the remote host and, separately, upload the large
# model weights fast by splitting them into chunks and pushing the chunks over
# several parallel SSH streams (then reassembling on the remote).
#
# Why parallel:
#   A single TCP/SSH stream to this host tops out at ~380 kB/s because of the
#   bandwidth-delay product at ~22 ms RTT (window-limited, NOT a hard cap).
#   Measured: 1 stream ~377 kB/s, 4 streams ~1348 kB/s aggregate (~3.5x).
#   So we split the 4 GB weights file and upload the pieces concurrently.
#
# OpenSSH scp uses SFTP by default.  Its default 64 * 32 KiB request window can
# also limit a high-bandwidth, high-latency stream.  The explicit -X settings
# below make its per-stream window 64 MiB, while parallel chunks fill multiple
# TCP flows.  Tune the environment variables below instead of editing this file:
#   CONCURRENCY=24 SFTP_REQUESTS=256 SFTP_BUFFER=261120 ./transfer_repo.sh
#   (261120 = 255 KiB: leave SFTP protocol-header room below the 256 KiB
#    OpenSSH server message cap.)
#
# Phases:
#   1. rsync the code tree (small) -- mirrors, excluding .git/.vscode and the
#      big static dirs (models/Qwen, train_dataset) which are handled separately.
#   2. Upload either model weights or the dataset as parallel SFTP chunks,
#      reassemble them remotely, and verify their checksum.
#
# Usage:
#   ./transfer_repo.sh                       # upload model weights (default)
#   TRANSFER_MODE=dataset ./transfer_repo.sh # archive and upload train_dataset
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
REMOTE_USER="titoffifee"
REMOTE_HOST="sdc-sim-gpu.sas.yp-c.yandex.net"
REMOTE_DIR="~/quality_control"

# Faster SSH cipher (AES-NI accelerated), no compression.  IPQoS=throughput
# avoids the interactive-traffic DSCP class on networks that honour it.
CIPHER="${CIPHER:-aes128-gcm@openssh.com}"
SSH_CMD="ssh -c ${CIPHER} -o Compression=no -o IPQoS=throughput"

# CHUNKS       : pieces to split the large file into (more = finer retries).
# CONCURRENCY  : simultaneous uploads. Raise gradually only if the remote
#                permits it; too many new connections can trip sshd limits.
# RAMP_DELAY   : seconds between opening uploads. It prevents sshd MaxStartups
#                from rejecting a large initial connection burst.
# RETRIES      : per-chunk retry attempts on transient failures.
# SFTP_*       : per-scp SFTP pipeline. Their product is the application-level
#                in-flight window; 256 * 255 KiB is about 64 MiB per upload
#                stream. Keep the buffer below 256 KiB: OpenSSH adds protocol
#                headers, and exactly 262144 causes "Outbound message too long".
CHUNKS="${CHUNKS:-256}"
CONCURRENCY="${CONCURRENCY:-32}"
RAMP_DELAY="${RAMP_DELAY:-0.20}"
RETRIES="${RETRIES:-6}"
SFTP_REQUESTS="${SFTP_REQUESTS:-256}"
SFTP_BUFFER="${SFTP_BUFFER:-261120}"

# Transfer selection. Dataset mode creates one uncompressed tar archive before
# upload. This avoids per-file SSH/SFTP overhead for the many image files.
TRANSFER_MODE="${TRANSFER_MODE:-weights}"
WEIGHTS_REL="models/Qwen/Qwen3-VL-Embedding-2B/model.safetensors"
DATASET_REL="train_dataset"

# Local repo = directory where this script lives.
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# ---------------------------------------------------------------------------
# Phase 1: mirror the code tree (fast, small).
# ---------------------------------------------------------------------------
echo ">>> [1/2] Syncing code tree -> ${REMOTE}:${REMOTE_DIR}/"
echo ">>> Remote is MIRRORED (extra files deleted), except excluded paths."

${SSH_CMD} "${REMOTE}" "mkdir -p ${REMOTE_DIR}"

# --delete removes remote files absent locally. Excluded paths are left as-is
# on the remote (we do NOT use --delete-excluded), so the weights/dataset that
# are already there are preserved and never uploaded by rsync.
rsync -avh --progress \
    --delete \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='models/Qwen' \
    --exclude='train_dataset' \
    -e "${SSH_CMD}" \
    "${LOCAL_DIR}/" \
    "${REMOTE}:${REMOTE_DIR}/"

# ---------------------------------------------------------------------------
# Phase 2: upload one large artifact via split + parallel streams.
# ---------------------------------------------------------------------------
DATASET_ARCHIVE_LOCAL=""
case "${TRANSFER_MODE}" in
    weights)
        LOCAL_WEIGHTS="${LOCAL_DIR}/${WEIGHTS_REL}"
        UPLOAD_REL="${WEIGHTS_REL}"
        ARTIFACT_LABEL="weights"
        ;;
    dataset)
        LOCAL_DATASET_DIR="${LOCAL_DIR}/${DATASET_REL}"
        if [[ ! -d "${LOCAL_DATASET_DIR}" ]]; then
            echo "!!! Dataset directory not found: ${DATASET_REL}" >&2
            exit 1
        fi
        DATASET_ARCHIVE_LOCAL="$(mktemp --suffix=.tar)"
        echo ">>> [2/2] Archiving ${DATASET_REL} locally (no compression: images are already compressed)..."
        tar -C "${LOCAL_DIR}" -cf "${DATASET_ARCHIVE_LOCAL}" "${DATASET_REL}"
        LOCAL_WEIGHTS="${DATASET_ARCHIVE_LOCAL}"
        UPLOAD_REL=".transfer_${DATASET_REL}.tar"
        ARTIFACT_LABEL="dataset archive"
        ;;
    *)
        echo "!!! Unknown TRANSFER_MODE=${TRANSFER_MODE}; use weights or dataset." >&2
        exit 2
        ;;
esac

if [[ ! -f "${LOCAL_WEIGHTS}" ]]; then
    echo ">>> [2/2] No local ${ARTIFACT_LABEL} at ${LOCAL_WEIGHTS}; skipping upload."
    echo ">>> Done."
    exit 0
fi

REMOTE_WEIGHTS_DIR="${REMOTE_DIR}/$(dirname "${UPLOAD_REL}")"
REMOTE_WEIGHTS="${REMOTE_DIR}/${UPLOAD_REL}"
BASENAME="$(basename "${UPLOAD_REL}")"

 echo ">>> [2/2] Uploading ${ARTIFACT_LABEL} in ${CHUNKS} chunks, ${CONCURRENCY} streams at a time"
echo ">>> Transport: scp/SFTP with ${SFTP_REQUESTS} requests x ${SFTP_BUFFER} bytes per stream; cipher=${CIPHER}; IPQoS=throughput"

# Idempotency: if the remote file already exists with the same checksum, skip.
LOCAL_SUM="$(sha256sum "${LOCAL_WEIGHTS}" | awk '{print $1}')"
REMOTE_SUM="$(${SSH_CMD} "${REMOTE}" "sha256sum ${REMOTE_WEIGHTS} 2>/dev/null | awk '{print \$1}'" || true)"
if [[ "${LOCAL_SUM}" == "${REMOTE_SUM}" ]]; then
    echo ">>> Remote weights already match (sha256=${LOCAL_SUM}); nothing to do."
    echo ">>> Done."
    exit 0
fi

# Work in a local temp dir for the chunks; clean up on exit.
TMPDIR_LOCAL="$(mktemp -d)"
REMOTE_TMP="${REMOTE_WEIGHTS_DIR}/.upload_${BASENAME}"
MONITOR_PID=""
cleanup() {
    [[ -n "${MONITOR_PID}" ]] && kill "${MONITOR_PID}" 2>/dev/null || true
    rm -rf "${TMPDIR_LOCAL}"
    [[ -n "${DATASET_ARCHIVE_LOCAL}" ]] && rm -f "${DATASET_ARCHIVE_LOCAL}"
    ${SSH_CMD} "${REMOTE}" "rm -rf ${REMOTE_TMP}" 2>/dev/null || true
}
trap cleanup EXIT

# Compute chunk size so we end up with exactly CHUNKS pieces.
FILESIZE="$(stat -c '%s' "${LOCAL_WEIGHTS}")"
CHUNK=$(( (FILESIZE + CHUNKS - 1) / CHUNKS ))

echo ">>> Splitting ${FILESIZE} bytes into ${CHUNKS} chunks of ~${CHUNK} bytes..."
split -b "${CHUNK}" -d -a 3 "${LOCAL_WEIGHTS}" "${TMPDIR_LOCAL}/chunk_"

# Prepare remote dirs.
${SSH_CMD} "${REMOTE}" "mkdir -p ${REMOTE_WEIGHTS_DIR} ${REMOTE_TMP}"

# --- Background progress monitor ---
# Polls the remote temp dir every few seconds and prints total bytes received,
# percentage of the full file, and an instantaneous rate. Runs until stopped.
MONITOR_INTERVAL=3
monitor() {
    local prev_bytes=0 prev_time
    prev_time="$(date +%s)"
    while true; do
        local cur_bytes now dt db rate pct
        cur_bytes="$(${SSH_CMD} "${REMOTE}" \
            "du -sb ${REMOTE_TMP} 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo 0)"
        cur_bytes="${cur_bytes:-0}"
        now="$(date +%s)"
        dt=$(( now - prev_time )); (( dt < 1 )) && dt=1
        db=$(( cur_bytes - prev_bytes ))
        rate=$(( db / dt ))
        pct=$(( FILESIZE > 0 ? cur_bytes * 100 / FILESIZE : 0 ))
        printf "\r>>> progress: %6d MB / %6d MB (%3d%%)  ~%5d kB/s   " \
            "$(( cur_bytes / 1048576 ))" "$(( FILESIZE / 1048576 ))" \
            "${pct}" "$(( rate / 1024 ))"
        prev_bytes="${cur_bytes}"; prev_time="${now}"
        sleep "${MONITOR_INTERVAL}"
    done
}
monitor &
MONITOR_PID="$!"

# Upload one chunk with retries + exponential backoff. Skips a chunk that is
# already fully present on the remote (lets you re-run to resume).
upload_chunk() {
    local chunk="$1" name attempt local_sz remote_sz
    name="$(basename "${chunk}")"
    local_sz="$(stat -c '%s' "${chunk}")"
    for (( attempt = 1; attempt <= RETRIES; attempt++ )); do
        remote_sz="$(${SSH_CMD} "${REMOTE}" \
            "stat -c '%s' ${REMOTE_TMP}/${name} 2>/dev/null" 2>/dev/null || echo 0)"
        if [[ "${remote_sz}" == "${local_sz}" ]]; then
            return 0
        fi
        if scp -c "${CIPHER}" -q \
            -o Compression=no -o IPQoS=throughput \
            -X "nrequests=${SFTP_REQUESTS}" \
            -X "buffer=${SFTP_BUFFER}" \
            "${chunk}" "${REMOTE}:${REMOTE_TMP}/${name}"; then
            return 0
        fi
        # Jitter stops rejected uploads from reconnecting in the same burst.
        sleep $(( attempt * 2 + RANDOM % 3 ))
    done
    echo "!!! chunk ${name} failed after ${RETRIES} attempts" >&2
    return 1
}

# Bounded worker pool: keep at most CONCURRENCY uploads in flight so we stay
# under the remote sshd connection limit. `wait -n` returns as soon as any one
# finishes, then we launch the next chunk.
echo ">>> Uploading chunks (${CONCURRENCY} at a time, ${RETRIES} retries each, ${RAMP_DELAY}s connection ramp)..."
fail=0
running=0
for chunk in "${TMPDIR_LOCAL}"/chunk_*; do
    upload_chunk "${chunk}" &
    running=$(( running + 1 ))
    # Opening 32 SCP sessions in one scheduler tick triggers sshd MaxStartups.
    # This one-time ramp is negligible beside a multi-gigabyte transfer.
    sleep "${RAMP_DELAY}"
    if (( running >= CONCURRENCY )); then
        wait -n || fail=1
        running=$(( running - 1 ))
    fi
done

# Wait for the remaining in-flight uploads from the worker pool.
while (( running > 0 )); do
    wait -n || fail=1
    running=$(( running - 1 ))
done

# Stop the progress monitor and finish its line.
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
printf "\n"

if [[ "${fail}" -ne 0 ]]; then
    echo "!!! One or more chunk uploads failed." >&2
    exit 1
fi

# Reassemble on the remote (chunks sort lexicographically = correct order).
echo ">>> Reassembling on remote..."
${SSH_CMD} "${REMOTE}" "cat ${REMOTE_TMP}/chunk_* > ${REMOTE_WEIGHTS}"

# Verify integrity end-to-end.
echo ">>> Verifying checksum on remote..."
REMOTE_SUM="$(${SSH_CMD} "${REMOTE}" "sha256sum ${REMOTE_WEIGHTS} | awk '{print \$1}'")"
if [[ "${LOCAL_SUM}" != "${REMOTE_SUM}" ]]; then
    echo "!!! Checksum mismatch! local=${LOCAL_SUM} remote=${REMOTE_SUM}" >&2
    exit 1
fi

if [[ "${TRANSFER_MODE}" == "dataset" ]]; then
    echo ">>> Extracting verified dataset archive on remote..."
    ${SSH_CMD} "${REMOTE}" "rm -rf ${REMOTE_DIR}/${DATASET_REL} && tar -C ${REMOTE_DIR} -xf ${REMOTE_WEIGHTS} && rm -f ${REMOTE_WEIGHTS}"
fi

 echo ">>> ${ARTIFACT_LABEL^} verified OK (sha256=${LOCAL_SUM})."
echo ">>> Done. Remote '${REMOTE_DIR}' is up to date."
