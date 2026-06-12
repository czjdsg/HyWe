# Vendored clash (USA topology)

Bundled copy of the USA-rule clash binary used by HybridWebAgent's
inference and synth3 pipelines.

| File | Purpose |
| --- | --- |
| `clash` | static linux/amd64 binary (Clash 2023.08.17) |
| `config.yaml` | proxy + rule + DNS config (verbatim copy of the USA topology used in production) |
| `Country.mmdb` | MaxMind GeoIP database used by the `GEOIP` rule |
| `clash_ctl.sh` | start / stop / status helper (tmux preferred, nohup fallback) |
| `cache.db` | DNS cache (auto-created on first run; gitignored) |

## Ports

| Port | Service |
| --- | --- |
| `127.0.0.1:8990` | HTTP proxy (set `TASK_PROXY_URL=http://127.0.0.1:8990`) |
| `127.0.0.1:8991` | SOCKS5 proxy |
| `127.0.0.1:9090` | External controller API |

These are pinned in `config.yaml` — if any of them collides with another
service on the host, edit `config.yaml` (`port` / `socks-port` /
`external-controller`) before starting.

## Usage

```bash
bash vendor/clash/clash_ctl.sh start
bash vendor/clash/clash_ctl.sh status
bash vendor/clash/clash_ctl.sh stop
```

After it's up, point the inference framework at it:

```bash
export TASK_PROXY_URL=http://127.0.0.1:8990
```

`configs/inference.env` already sets this by default — sourcing it picks
up the proxy automatically.

## Notes

- The binary is linux/amd64. For other architectures, replace `clash`
  with the matching official build from <https://github.com/Dreamacro/clash/releases>
  (or its mihomo successor) and keep `config.yaml` / `Country.mmdb` as-is.
- `config.yaml` contains live proxy credentials. The repo is private, so
  they're checked in verbatim — if you flip the repo to public, scrub
  this file first.
- `cache.db` is regenerated on every cold start; nothing in it is
  load-bearing.
