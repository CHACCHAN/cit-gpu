# 新習志野 FreeToken

`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` を A4500 ごとに FreeToken 1 worker で動かす。SGLang Model Gateway がポート5050で worker に round robin で振り分け、既存の Cloudflare Tunnel が Gateway を公開する。worker の API ポートは `5051, 5053, 5055, 5057, 5059`。各ポートの次の番号は FreeToken の内部通信に使われる。

モデルの `text_config.max_position_embeddings` は262144。`--max-seq-len-override`、`--num-tokens`、`--kv-reserve-tokens` を262144に指定し、YaRN は使わない。A4500 20 GBでは `--text-model-only` と最小512枠の MoE cache、prefill overlap 無効化が必要。画像入力は受け付けない。QSA の KV cache は FreeToken の現行経路では BF16で、約6.19 GiB/worker。8bit KV を指定する CLI はない。

## 実測と制約

- 指定 checkpoint は約126 GiB。home の80 GB制限には入らないため、`stage_model.py` が計算ノードの `/tmp` に取得する。gpu03 の空きは初回検証時に778 GiB。別ノードへの移動や scratch 消去後は再取得が必要。
- FreeToken は PLE 表約47.7 GiBを disk backend で読む。expert bank 約63.3 GiBは各 worker のホストメモリに置く。5 worker 準備完了時の Slurm cgroup 使用量は約331 GiB / 480 GiB。
- 1 worker の `--moe-cache-size 512`、`--max-prefill-length 1024` で、262144 tokens の KV cache を確保して `/health=ok`、短い推論に成功（job 16186）。`/v1/models` の `max_model_len` と `context_length` は262144。ポート5050の Gateway と `https://gpgpu.cc-chacchan.com/v1/chat/completions` でも認証付きで `OK.` を確認。
- 自動 MoE cache は993枠を割り当て、CUDA graph の準備中に VRAM OOM となった。512枠では起動した。最小枠なので cache miss と PCIe転送が増える可能性がある。
- 5 worker は job 16189 で全員 `/health=ok`。8,192 tokens と65,536 tokens の実リクエストはそれぞれ約36秒、216秒で完了。262,000 tokens を狙った実リクエストは API の実測 `prompt_tokens=261988`、`completion_tokens=8`、985秒で完了し、入力切り詰めはなかった。

## セットアップ

FreeToken は commit `0d652e73a452d014ac5441a15baa75348e9fcb0a` で検証している。ホスト既定の nvcc は12.2なので、ビルドと実行には SGLang venv の CUDA 13 toolkit を使う。

```bash
cd freetoken
python3 -m venv .venv
python3 -m venv .router-venv
.router-venv/bin/pip install --no-cache-dir sglang-router==0.3.2
FT_CUDA_HOME="$PWD/../SGLang/.venv/lib/python3.12/site-packages/nvidia/cu13"
CUDA_HOME="$FT_CUDA_HOME" PATH="$FT_CUDA_HOME/bin:$PATH" \
  .venv/bin/pip install --no-cache-dir \
  'freetoken[accel] @ git+https://github.com/FlashML-org/FreeToken.git@0d652e73a452d014ac5441a15baa75348e9fcb0a'
cp .env.example .env
chmod 600 .env
# .env に FT_API_KEY（従来の LLAMA_API_KEY）と TUNNEL_TOKEN を設定
mkdir -p logs
sbatch shinnarashino
```

ジョブは5 worker を順に起動する。同じ checkpoint を5回同時に読むと node-local disk が極端に遅くなったため。全 worker が ready になってから Gateway を起動し、`verify_routing.py` が全5基への配信を確認してから Tunnel を起動する。`FT_REPLICAS=1` と `sbatch --gpus-per-task=1 --mem=160G shinnarashino` で1 GPU試験もできる。`python3 launch.py --dry-run` は起動引数を表示する。

Cloudflare の構成は従来の `gpgpu.cc-chacchan.com -> http://localhost:5050` のまま。ポート5050は認証付き Gateway、worker は localhost のみ。Python の既定 User-Agent からこの公開ホストを呼ぶと Cloudflare error 1010 になるが、ブラウザ相当の User-Agent では HTTP 200 と推論を確認した。

## 確認

`squeue -u "$USER"` でジョブIDを調べ、`logs/<jobid>.log` の `5 workers and Gateway ready` を確認する。各 worker の `/health` が `{"status":"ok"}` を返す。`srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 .venv/bin/python verify_routing.py` で5 worker への振り分けを確認する。`/v1/models` の表示だけでは長文の実処理を証明しないので、同様に `verify_context.py --target-tokens 262000` で長文テストする。

Slurm の research partition 自体には時間上限がないが、このアカウントの QOS により1ジョブ24時間が上限。起動に成功したジョブは `FT_AUTO_RENEW=1` のとき `--dependency=singleton` の後続ジョブを1件登録する。再起動中はモデルの読み込みに約13分かかり、公開APIもその間停止する。node-local の checkpoint はジョブの終了後も残る場合があるが永続ではない。`launch.py` は見つからなければ `stage_model.py` で再取得する。永続運用と起動短縮には、管理者が書き込み可能な共有 scratch 領域への配置が望ましい。

仕様: [FreeToken CLI](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)、[対応モデル](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)。
