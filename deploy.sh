#!/usr/bin/env bash
# Deploy to a fresh Ubuntu VM (tested target: Oracle Cloud Ampere A1, arm64, >= 4 GB RAM):  ./deploy.sh ubuntu@<vm-ip>
# Also open TCP 80 in the VCN security list (Oracle console) — this script handles the VM's own iptables.
set -euo pipefail
HOST="$1"; cd "$(dirname "$0")"
rsync -az --delete --exclude .venv --exclude .hydradb --exclude __pycache__ --exclude data/raw --exclude '.git' \
  --include 'data/' --include 'data/traces_slim.json' --include 'data/*.npy' --include 'data/results_2000_k10_n2.json' --include 'data/load.json' \
  --exclude 'data/*' ./ "$HOST":deja/
ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sudo sh; sudo usermod -aG docker "$USER"; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/deja && uv sync -q
sudo -E env "PATH=$PATH" BUCKET=deja ./hydra.sh   # sudo: fresh docker group membership needs a re-login otherwise
sudo tee /etc/systemd/system/deja.service >/dev/null <<UNIT
[Unit]
Description=Deja server
After=docker.service
[Service]
User=$USER
WorkingDirectory=$HOME/deja
Environment=OMP_NUM_THREADS=1 PORT=8090 PATH=$HOME/.local/bin:/usr/bin:/bin
ExecStart=$HOME/.local/bin/uv run python -u server.py
Restart=on-failure
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now deja && sudo systemctl restart deja
# Oracle Ubuntu images ship an iptables REJECT rule: allow 80 and redirect it to the app
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT
# REDIRECT runs in PREROUTING, so INPUT sees the rewritten port: allow 8090 too
sudo iptables -C INPUT -p tcp --dport 8090 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 5 -p tcp --dport 8090 -j ACCEPT
sudo iptables -t nat -C PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8090 2>/dev/null || sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8090
command -v netfilter-persistent >/dev/null && sudo netfilter-persistent save || true
echo "started; first boot loads the graph + held-out check (~2-3 min): journalctl -u deja -f"
REMOTE
