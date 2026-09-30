# FreeToken (新習志野 A4500 / 津田沼 H100)

以下は新習志野の記録。津田沼 (H100 NVL ×2、GPU 1 枚 8 並列) は末尾の「津田沼」を参照。

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

Cloudflare の構成は従来の `gpgpu.cc-chacchan.com -> http://localhost:5050` のまま。ポート5050は鍵を検証する `auth_proxy.py`（下の「認証」）、worker は localhost のみ。Python の既定 User-Agent からこの公開ホストを呼ぶと Cloudflare error 1010 になるが、ブラウザ相当の User-Agent では HTTP 200 と推論を確認した。

## 確認

`squeue -u "$USER"` でジョブIDを調べ、`logs/<jobid>.log` の `5 workers and Gateway ready` を確認する。各 worker の `/health` が `{"status":"ok"}` を返す。`srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 .venv/bin/python verify_routing.py` で5 worker への振り分けを確認する。`/v1/models` の表示だけでは長文の実処理を証明しないので、同様に `verify_context.py --target-tokens 262000` で長文テストする。

Slurm の research partition 自体には時間上限がないが、このアカウントの QOS により1ジョブ24時間が上限。起動に成功したジョブは `FT_AUTO_RENEW=1` のとき `--dependency=singleton` の後続ジョブを1件登録する。再起動中はモデルの読み込みに約13分かかり、公開APIもその間停止する。node-local の checkpoint はジョブの終了後も残る場合があるが永続ではない。`launch.py` は見つからなければ `stage_model.py` で再取得する。永続運用と起動短縮には、管理者が書き込み可能な共有 scratch 領域への配置が望ましい。

仕様: [FreeToken CLI](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)、[対応モデル](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)。

## 津田沼 (H100 NVL ×2、GPU 1 枚 8 並列)

同じモデルを H100 NVL 94 GB ごとに 1 worker、`--max-running-requests 8` で動かす。research の上限 (GPU 2 / CPU 16 / mem 256 GB) から worker は 2 基、合計 16 並列。公開 URL は従来の MiMo と同じ `https://gpgpu2.cc-chacchan.com`。

```bash
cd freetoken   # または リポジトリ直下からでもよい (ジョブファイルが cd する)
# 導入は上の「セットアップ」と同じ。加えて SGLang の venv に3本リンクが要る (下記)
cp .env.example .env && chmod 600 .env   # 津田沼は下の設定にする。鍵と Tunnel は llama/.env と同じ
sbatch tsudanuma
```

`.env` の津田沼向け設定:

```
FT_REPLICAS=2
FT_JOB_SCRIPT=tsudanuma
FT_MAX_RUNNING_REQUESTS=8
FT_NUM_TOKENS=2097152      # 8 * 262144。全枠が 256K を持てる
FT_MEMORY_RATIO=0.92
FT_MOE_CACHE_AUTO=1
FT_MAX_PREFILL_LENGTH=4096
FT_DISABLE_MOE_PREFILL_OVERLAP=0
```

### 導入で必要だった CUDA 側の補正

SGLang の venv の CUDA 13 toolkit (pip 版) は JIT ビルドに必要なリンクが足りない。`.venv` は git 管理外なので手で足す。

```bash
C=SGLang/.venv/lib/python3.12/site-packages/nvidia/cu13
ln -s libcudart.so.13 $C/lib/libcudart.so                    # FreeToken のビルド
ln -s lib $C/lib64                                           # FreeToken の JIT
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $C/lib/libcuda.so   # FlashInfer の JIT (計算ノードでのみ有効なリンク)
```

nvcc は 13.3、ヘッダは 13.0 で、FlashInfer の JIT が拒否する。`launch.py` は `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` を設定して回避している。

### 8 並列の実測 (2026-09-30、H100 NVL 1 枚)

- 津田沼の checkpoint は計算ノードの `/tmp` (空き約560 GiB) に取得する。約4分。ジョブ内で `stage_model.py` が実行する。
- 8 並列 × 262144 の KV は 2097152 tokens = 49.5 GiB。MoE cache を自動 (8076枠) にすると VRAM は約87 GiB 使用 (89,015 MiB、空き約7.2 GiB)。512枠に固定すると約67 GiB で、8並列の短い入力は約2.6倍遅い (5.5 → 14.2 tok/s/リクエスト、prefill 1024 のとき)。
- 新習志野の `--max-prefill-length 1024` と MoE prefill overlap 無効は H100 では遅い。4096 と overlap 有効にすると 8 × 32K の全体時間が 476 秒から 135 秒になった。
- 1 並列のデコードは約61.5 tok/s。8 並列の短い入力は 1 リクエスト約34.7 tok/s (合計約199 tok/s)。
- 8 本の同時リクエストで各 250,088 tokens の入力を投げ、全て完走 (1006 秒、切り詰めなし、完走後も `/health=ok`)。prefill は 1 本ずつ直列で約125秒/本、8 本目の最初のトークンは約1000秒後。
- 自動 MoE cache は、A4500 20 GB では CUDA graph 準備中に OOM したが、H100 94 GB では起動する。

### 1 GPU あたりのスロット数を増やした場合 (2026-09-30)

KV は 2097152 tokens の共有プールで、スロット数を増やしても VRAM はほぼ変わらない (約87 GiB)。mamba スロットは最大並列数の6倍が確保される (8 で48、16 で96、32 で192)。CUDA graph は最大並列数まで取得される。ただし最大並列数が大きい worker ほど、同じ本数を流してもデコードが遅い。

| 最大並列数 | 実際の同時本数 | 合計 tok/s | 1 リクエスト tok/s |
|---|---|---|---|
| 8 | 8 | 約199 | 34.7 |
| 16 | 8 | 約156 | 25 |
| 16 | 16 | 約177 | 13.6 |
| 32 | 8 | 約48 | 6.5 |
| 32 | 32 | 約86 | 2.9 |

16 並列 × 32K トークンの同時入力は 264 秒で完走 (prefill は 8 並列と同じく 1 本約16秒)。全スロットに 256K を持たせるには KV が足りないため (16 × 256K = 4194304 tokens > 2097152)、その場合は最大並列数 8 のままにする。合計速度を優先するなら 8、セッション数を優先するなら 16。32 は勧めない。

### 運用上の注意

- Gateway は `cache_aware` で 2 worker に振る (`FT_ROUTING_POLICY`、既定は round_robin)。同じ長い履歴は同じ worker に送られ、prefix cache が効く。30K tokens の共有文書で 1 回目 15.4 秒、2 回目以降 1.4 秒。偏りが閾値を超えると空いている worker に回る。1 worker の 9 件目以降は worker 内の待ち行列に入る。
- 長い入力の prefill は worker 内で直列。長文が混じると同じ worker の他の要求の最初のトークンが遅れる。
- MiMo のモデル・LoRA は削除済み。

### 認証

SGLang Gateway の `control_plane_api_keys` は管理 API しか守らず、`/v1/*` は鍵なしで通る。FreeToken の worker にも鍵の指定はない。そこで公開ポート 5050 は `auth_proxy.py` が受け、`Authorization: Bearer <FT_API_KEY>` が正しい要求だけを内部の Gateway (`FT_GATEWAY_PORT`、既定 5049) に流す。`/v1/*` 以外は通さない (`/workers` は 404)。リクエストの中身はログに出さない。

`launch.py` はトンネルを起動する前に自己検査を行い、鍵なし・誤った鍵が 401、正しい鍵が 200、管理 API が閉じていることを確認する。通らなければ起動を中止する。津田沼で公開して、鍵なしと誤った鍵が全パスで 401、正しい鍵が `/v1/*` で 200 になることも確認した。新習志野も同じ `run_gateway.py` を使っていたため、次の起動から同じ保護が入る（それまでは鍵なしで推論できる可能性がある）。

### ジョブの更新

計算ノード上の `sbatch` は動かない (`No module named 'encodings'`)。`launch.py` のジョブ内での後続ジョブ登録は失敗するので、津田沼は `FT_AUTO_RENEW=0` にし、ログインノードから登録する。

```bash
sbatch --dependency=afterany:<現在のジョブID> tsudanuma
```

### メモリ使用量 (最大並列 8 × 2 worker、実測)

- VRAM: 1 枚あたり 89,015 MiB (約87 GiB / 93.6 GiB)。
- ホスト: worker プロセスの RSS は各約65 GiB (計約131 GiB)。ジョブの cgroup は 198 GiB、ピーク 202 GiB (制限 256 GiB)。大半は expert bank の共有メモリとページキャッシュ。
- 起動時間: checkpoint が page cache に残っていれば約7分、消えていれば expert bank の読み込みだけで約21分。2 worker の同時起動は速くならなかったので 1 基ずつ起動する。
- プレフィルのチャンク長 4096 → 8192 は 8 × 32K で約3%短縮のみで、採用しない。
