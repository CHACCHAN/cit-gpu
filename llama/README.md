# llama.cpp 配信

llama-server で OpenAI 互換 API を提供する。起動処理は両サイト共通。

| サイト | ジョブ | モデル | 手順 |
|---|---|---|---|
| 津田沼 (H100 NVL ×2) | `mimo.tsudanuma` | MiMo V2.6 Flash RL Q3_K + LoRA、262144 ×4 slot | [docs/tsudanuma.md](docs/tsudanuma.md) |
| 新習志野 (A4500 ×5) | `bonsai.shinnarashino` | Ternary Bonsai 2 27B PQ2_0、262144 ×8 slot | [docs/shinnarashino.md](docs/shinnarashino.md) |

- `src/launch.py`: llama-server と Cloudflare Tunnel の起動・終了 (port 5050、API key は `.secrets/api-keys`)
- `src/build.py`: llama.cpp を commit 固定でビルド。CUDA runtime を同梱する
- `src/fetch.py`: Hugging Face から revision 固定で取得し SHA-256 を検証
- `src/verify.py`: 実 API の検証 (`--slots` / `--slot-ctx` / `--body`)。`src/verify_lora.py` は MiMo 専用

ジョブファイルにはビルド・モデル・llama-server 引数だけを書く。
`bash <ジョブ> --dry-run` で起動コマンドを確認できる。
`models/` `adapters/` `llama.cpp*/` `.secrets/` `logs/` は Git 管理外。

クラスタ共通: sbatch ラッパは終了コードを伝えない (`logs/<jobid>.err` の `ERROR:` を見る)。
home は計算ノードと同期されている。演習モード中は research ジョブが強制終了・requeue される。
