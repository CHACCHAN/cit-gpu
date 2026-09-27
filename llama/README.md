# llama.cpp 配信

llama-server で OpenAI 互換 API を提供する。コマンドはすべて `llama/` で実行する。

| サイト | ジョブ | モデル | 公開URL |
|---|---|---|---|
| 津田沼 (H100 NVL ×2) | `tsudanuma/mimo` | MiMo V2.6 Flash RL Q3_K + LoRA、262144 ×6 slot | `https://gpgpu2.cc-chacchan.com` |
| 新習志野 (A4500 ×5) | `shinnarashino/bonsai` | Ternary Bonsai 2 27B PQ2_0、GPU 1 枚ずつ 163840 ×2 slot、計10並列 | `https://gpgpu.cc-chacchan.com` |

セットアップと運用は [津田沼](tsudanuma/README.md) / [新習志野](shinnarashino/README.md) を参照。

## SGLang Gateway (両サイト共通、初回だけ)

SGLang Model Gateway が公開ポート 5050 を受け、llama-server の worker に推論を振り分ける。
公開 `/props`、`/slots`、`/metrics`、`/lora-adapters` は 404。
`/v1/models` は鍵なしで 200 を返すが、ID は `unknown`。
管理用 `/workers` は鍵なし 401、既存の推論用鍵でも 403。
席数は利用側で Bonsai 10、MiMo 6 に固定する。推論には既存の `LLAMA_API_KEY` が必要。

## 混雑時

Hermes の `relay-gpu-gate` は Bonsai 10 件、MiMo 6 件を超える依頼を Hermes 側で待たせる。
Gateway 単体の上限は同時 64 件、待機列 100 件、待機期限 600 秒。
この上限は worker の席数とは連動しない。公開 API を直接呼ぶ場合や、cache-aware の振り分けが
一台に偏る場合は、llama-server 側で席待ちが起きる。長時間応答が始まらなければ
Cloudflare のタイムアウトに達する可能性がある。

`llama/` で Gateway をインストールする。

```bash
python3 -m venv .router-venv
.router-venv/bin/pip install 'sglang-router==0.3.2'
```

`.env` に `LLAMA_API_KEY` と `TUNNEL_TOKEN` を設定する。ジョブのログは投入ディレクトリの `logs/` に出る。

## 運用

ジョブは 24 時間で終了する。継続する場合は `sbatch --dependency=singleton <ジョブ>` で
同名の後続ジョブを 1 件予約する。`bash <ジョブ> --dry-run` で起動コマンドを確認できる。
検証スクリプトは worker のローカルポート 5051 以降を使う。Gateway は 5050、Cloudflare Tunnel はそのポートへ接続する。

プロンプトには個人の記録が含まれるため、llama-server の詳細ログ (`-v`) を有効にしない。
API 鍵と Tunnel token は `.env` (権限 600) から読み、ジョブの引数には含めない。
