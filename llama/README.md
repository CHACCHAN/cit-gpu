# llama.cpp 配信

llama-server で OpenAI 互換 API を提供する。コマンドはすべて `llama/` で実行する。

| サイト | ジョブ | モデル | 公開URL |
|---|---|---|---|
| 津田沼 (H100 NVL ×2) | `tsudanuma/mimo` | MiMo V2.6 Flash RL Q3_K + LoRA、393216 ×4 slot | `https://gpgpu2.cc-chacchan.com` |
| 新習志野 (A4500 ×5) | `shinnarashino/bonsai` | Ternary Bonsai 2 27B PQ2_0、GPU 1 枚ずつ 262144 ×5 slot | `https://gpgpu.cc-chacchan.com` |

手順とサイト固有の注意は各フォルダの README.md。

## Caddy (両サイト共通、初回だけ)

公式リリースの static binary を checksum を確かめて `bin/caddy` に置く。

```bash
mkdir -p bin && V=2.11.4 && (cd bin &&
    curl -sSLO https://github.com/caddyserver/caddy/releases/download/v$V/caddy_${V}_linux_amd64.tar.gz &&
    curl -sSLO https://github.com/caddyserver/caddy/releases/download/v$V/caddy_${V}_checksums.txt &&
    grep " caddy_${V}_linux_amd64.tar.gz$" caddy_${V}_checksums.txt | sha512sum -c - &&
    tar -xzf caddy_${V}_linux_amd64.tar.gz caddy && rm caddy_${V}_*)
```

## 個人情報

プロンプトには個人の記録が含まれる。llama-server を `-v` など詳しいログで動かさない
(既定のログは処理時間と token 数だけ)。Caddy はアクセスログを出さない。

- `common/launch.py`: llama-server・Caddy・Cloudflare Tunnel の起動と終了。llama-server は 127.0.0.1:5051 から、
  Caddy (`bin/caddy` + `common/Caddyfile`) が port 5050 で受ける。`--replicas N` で GPU を N 組に分けて N 個起動する
- `common/Caddyfile`: cookie で同じクライアントを同じ llama-server に固定 (新規・固定先が満杯なら処理中の少ない方)。
  slot が全部埋まった llama-server には送らず、送り先が無ければ最大 45 秒待ってから 503 + `Retry-After: 2`
  (45 秒 + 最初の `:` まで最大 30 秒で Cloudflare の 100 秒に収まる)。`/slots` `/metrics` は Cloudflare 経由では 403 (LAN からは見える)
- `common/build.py`: llama.cpp を commit 固定でビルド。CUDA runtime を同梱する
- `common/fetch.py`: Hugging Face から revision 固定で取得し SHA-256 を検証
- `common/verify.py`: 実 API の検証 (`--slots` / `--replicas` / `--slot-ctx` / `--body`)。`tsudanuma/verify_lora.py` は MiMo 専用
- `.env` (600): `LLAMA_API_KEY` と `TUNNEL_TOKEN`。`TUNNEL_TOKEN` が無ければ Tunnel は起動しない。雛形は `.env.example`

ジョブファイルにはビルド・モデル・llama-server 引数だけを書く。
`llama/` からでも各サイトのフォルダからでも `sbatch` できる。ログは投入したディレクトリの `logs/` に出る (先に `mkdir -p logs`)。
`bash <ジョブ> --dry-run` で起動コマンドを確認できる。
`models/` `adapters/` `llama.cpp*/` `bin/` `.env` `logs/` は Git 管理外。

クラスタ共通: sbatch ラッパは終了コードを伝えない (`logs/<jobid>.err` の `ERROR:` を見る)。
home は計算ノードと同期されている。演習モード中は research ジョブが強制終了・requeue される。
