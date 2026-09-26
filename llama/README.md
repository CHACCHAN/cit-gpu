# llama.cpp 配信

llama-server で OpenAI 互換 API を提供する。コマンドはすべて `llama/` で実行する。

| サイト | ジョブ | モデル | 公開URL |
|---|---|---|---|
| 津田沼 (H100 NVL ×2) | `tsudanuma/mimo.sbatch` | MiMo V2.6 Flash RL Q3_K + LoRA、524288 ×3 slot | `https://gpgpu2.cc-chacchan.com` |
| 新習志野 (A4500 ×5) | `shinnarashino/bonsai.sbatch` | Ternary Bonsai 2 27B PQ2_0、GPU 1 枚ずつ 262144 ×5 slot | `https://gpgpu.cc-chacchan.com` |

手順とサイト固有の注意は各フォルダの README.md。

- `common/launch.py`: llama-server と Cloudflare Tunnel の起動・終了 (port 5050)。
  `--replicas N` で GPU を N 組に分けて N 個起動し、Caddy (`bin/caddy` + `common/Caddyfile`) が 5050 で振り分ける
- `common/build.py`: llama.cpp を commit 固定でビルド。CUDA runtime を同梱する
- `common/fetch.py`: Hugging Face から revision 固定で取得し SHA-256 を検証
- `common/verify.py`: 実 API の検証 (`--slots` / `--replicas` / `--slot-ctx` / `--body`)。`tsudanuma/verify_lora.py` は MiMo 専用
- `.env` (600): `LLAMA_API_KEY` と `TUNNEL_TOKEN`。`TUNNEL_TOKEN` が無ければ Tunnel は起動しない。雛形は `.env.example`

ジョブファイルにはビルド・モデル・llama-server 引数だけを書く。
`bash <ジョブ> --dry-run` で起動コマンドを確認できる。
`models/` `adapters/` `llama.cpp*/` `bin/` `.env` `logs/` は Git 管理外。

クラスタ共通: sbatch ラッパは終了コードを伝えない (`logs/<jobid>.err` の `ERROR:` を見る)。
home は計算ノードと同期されている。演習モード中は research ジョブが強制終了・requeue される。
