#!/usr/bin/env bash
# Single-node HydraDB (open-source engine): RustFS (S3) + graph-node + graph-indexer.
# S3 is required in practice: the local-FS backend lacks conditional PUT, so a restarted node can't retake its write
# lease and the indexer can't publish index generations (reads then replay the whole WAL and crawl).
# BUCKET=<name> ./hydra.sh   picks a graph (bucket); a new name = empty graph.
set -euo pipefail
cd "$(dirname "$0")"
BUCKET="${BUCKET:-deja}"; KEY=deja; SECRET=deja-local-secret
mkdir -p .hydradb/s3 .hydradb/cache .hydradb/icache
[ -f .hydradb/auth-token ] || od -An -tx1 -N24 /dev/urandom | tr -d " \n" > .hydradb/auth-token
docker rm -f hydradb hydradb-indexer >/dev/null 2>&1 || true
if ! docker ps --format '{{.Names}}' | grep -qx hydradb-s3; then
  docker rm -f hydradb-s3 >/dev/null 2>&1 || true
  docker run -d --name hydradb-s3 --restart unless-stopped --user "$(id -u):$(id -g)" --network host -v "$PWD/.hydradb/s3:/data" \
    -e RUSTFS_ACCESS_KEY=$KEY -e RUSTFS_SECRET_KEY=$SECRET -e RUSTFS_ADDRESS=127.0.0.1:9000 -e RUSTFS_CONSOLE_ENABLE=false \
    rustfs/rustfs:1.0.4 /data >/dev/null
  sleep 3
fi
uv run -q --no-project --with boto3 python -c "
import boto3; s3 = boto3.client('s3', endpoint_url='http://127.0.0.1:9000', aws_access_key_id='$KEY', aws_secret_access_key='$SECRET', region_name='us-east-1')
if '$BUCKET' not in [b['Name'] for b in s3.list_buckets()['Buckets']]: s3.create_bucket(Bucket='$BUCKET')"
ENV=(-e CLOUD_PROVIDER=aws -e AWS_BUCKET_NAME="$BUCKET" -e AWS_DEFAULT_REGION=us-east-1 -e AWS_ALLOW_HTTP=true
  -e AWS_ENDPOINT=http://127.0.0.1:9000 -e AWS_ACCESS_KEY_ID=$KEY -e AWS_SECRET_ACCESS_KEY=$SECRET
  -e GRAPH_NAMESPACE=default -e GRAPH_ID=default -e GRAPH_CELL_ID=cell-0 -e GRAPH_CELLS=cell-0
  -e GRAPH_NODE_ID=node-0 -e GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 -e GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687
  -e GRAPH_AUTH_TOKEN_FILE=/data/auth-token -e GRAPH_ALLOW_PLAINTEXT=true -e RUST_MIN_STACK=33554432
  -e GRAPH_USAGE_REPORT_ENABLED=false -e GRAPH_WAL_FLUSH_INTERVAL_MS=5 -e GRAPH_INDEXER_DIRTY_POLL_INTERVAL_MS=1000
  -e GRAPH_MAX_CONCURRENT_HYDRATIONS=8 -e GRAPH_DATA_CACHE_BYTES=2147483648)
docker run -d --name hydradb --restart unless-stopped --user "$(id -u):$(id -g)" --network host \
  -v "$PWD/.hydradb:/data" "${ENV[@]}" -e GRAPH_DATA_CACHE_DIR=/data/cache \
  -e GRAPH_BOLT_ADDR=127.0.0.1:7687 -e GRAPH_HTTP_ADDR=127.0.0.1:8443 -e GRAPH_ADMIN_ADDR=127.0.0.1:9090 \
  ghcr.io/hydra-db/hydradb:0.4.0 >/dev/null
docker run -d --name hydradb-indexer --restart unless-stopped --user "$(id -u):$(id -g)" --network host \
  -v "$PWD/.hydradb:/data" "${ENV[@]}" -e GRAPH_DATA_CACHE_DIR=/data/icache -e GRAPH_ADMIN_ADDR=127.0.0.1:9091 \
  --entrypoint /usr/local/bin/graph-indexer ghcr.io/hydra-db/hydradb:0.4.0 >/dev/null
for i in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:9090/readyz && break; sleep 1; done
echo "HydraDB ready on bolt://127.0.0.1:7687 (bucket $BUCKET)"
