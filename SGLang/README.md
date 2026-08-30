# Qwen3.8-27B を SGLang で配信 (新習志野 / gpumng)

新習志野 GPGPU 1号機 (`gpumng.cle.it-chiba.ac.jp`, RTX A4500 20GB x4, TP=4) で
SGLang を動かし、OpenAI 互換 API を CodeAgent 向けに提供する。
津田沼 (`gpumng2`, H100 NVL x2) の DeepSeek-V4-Flash 構成からの移行。

ジョブファイル (`sglang.shinnarashino` / `sglang.tsudanuma`) を sbatch する。
`src/` は両サイト共通の汎用パイプライン (Python)。サイト差分 (必須変数・
環境変数・serve 追加引数) はジョブファイル側の bash で組み立てて流し込む:

```
python3 -u src/launch.py [--marker FILE]... [--dry-run] -- [serve 追加引数...]
```
設定は `.env` のみ。

| パス | 役割 |
|---|---|
| `sglang.shinnarashino` / `sglang.tsudanuma` | sbatch するジョブファイル。`#SBATCH` / .env の source / サイト固有の env と serve 引数 / launch.py の exec |
| `src/launch.py` | 汎用パイプライン: 検証 → Tunnel → venv → モデル取得 → serve。`--dry-run` で serve 直前まで流して引数を表示 |
| `src/tunnel.py` | Cloudflare Tunnel (`TUNNEL_TOKEN` が空ならスキップ) |
| `src/venvsetup.py` | venv 構築 / CUDA toolchain 補正 (`SG_CUDA_FIXUPS=1`) / ログノイズの後始末 (`patch_sarashina()` 含む) |
| `src/modelfetch.py` | モデル取得 |
| `src/serve.py` | 共通 serve 引数 + 起動ウォッチドッグ |
| `src/p2pdiag.py` ほか | 診断ツール (p2p_check / dist_smoke / sweep)。ジョブからは使わない |

## 使い方

```bash
cd SGLang
cp .env.example .env      # SG_API_KEY と TUNNEL_TOKEN を入れる
sbatch sglang.shinnarashino
tail -f logs/<jobid>.err  # sglang のログは stderr
```

初回は venv 作成 (約8分 / 8.9GB) とモデル取得 (20GB) が走る。
READY までの時間は triton の JIT キャッシュが温まっているかで大きく変わる。
コールド (venv 作り直し後など) は約14分 (実測 835s / job 14288、うち
prefill CUDA graph capture が 441s)。ウォームなら約3分 (185s / job 14299、
同 capture 26s)。`SG_STARTUP_TIMEOUT` はコールドを基準に取ること。
`The server is fired up and ready to roll!` が出れば稼働。停止は `scancel <jobid>`。

ジョブは `--partition=research` なので実行ノードは毎回変わる。
API のアドレスは `squeue -h -j <jobid> -o %N` で確認する。

## API

```bash
NODE=$(squeue -h -u $USER -o %N)
KEY=$(grep ^SG_API_KEY= .env | cut -d= -f2-)

curl -s -H "Authorization: Bearer $KEY" http://$NODE:5050/v1/models | python3 -m json.tool

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:5050/v1/chat/completions \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],
       "max_tokens":512}' | python3 -m json.tool
```

reasoning は `reasoning_content`、tool call は `message.tool_calls` で返る。
Prometheus メトリクスは `/metrics` (`--enable-metrics`)。

CodeAgent 側は `contextWindow: 262144` を指定してよい。
**ただし `max_tokens` は必ず入れること** (「既知の問題」参照)。

## 構成の要点

| 項目 | 値 | 理由 |
|---|---|---|
| checkpoint | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (20GB) | compressed-tensors W4A16 / group32 / 非対称 → Marlin。GDN・vision・lm_head は BF16 のまま |
| TP | 4 | heads 24 / kv_heads 4 / GDN 16:48 / FFN 17408 はいずれも 5 で割り切れない |
| context | 262144 (native) | `max_position_embeddings`。YaRN 不使用 |
| attention | triton | flashinfer は動くが、SGLang 自身が Qwen3_5 系の既定に triton を選んでいる |
| KV | bf16 / 645,924 token | checkpoint が `kv_cache_quant_algo` を宣言しないので auto=bf16。A4500 に FP8 テンソルコアは無い |
| GDN state | bfloat16 / 79 slot | fp32 の 153.9MB/slot に対し bf16 は 78.4MB/slot |
| HiCache | 40GB/rank = 139GB | `--hicache-size` は **rank ごとの GB**。TP4 なので総量は4倍 |

### venv の CUDA まわり (ここを外すと起動しない)

`src/venvsetup.py` が venv 作成後にやっている3点 (`SG_CUDA_FIXUPS=1` のときだけ)。
素のノードではどれも必須。コンテナには /usr/local/cuda が一式あるので不要
(津田沼は既定 0 のまま)。

1. **`nvidia-cuda-nvcc` を 13.0.88 に固定する。**
   素の依存解決では nvcc が 13.3.73 まで上がる一方 `nvidia-cuda-runtime` は
   13.0.96 のまま残り、flashinfer 同梱 cccl の
   `cuda_toolkit.h: "CUDA compiler and CUDA toolkit headers are incompatible"`
   で全 JIT カーネルがビルド不能になる。`cuda-toolkit 13.0.3.0` 自身が
   nvcc 13.0.88 / runtime 13.0.96 を組で要求しているので、その組に戻す。
2. **`$CUDA_HOME/lib64` を symlink farm として作る。**
   pip の wheel は `lib/libcudart.so.13` しか置かず `lib64/` も
   開発用 `libcudart.so` symlink も無いため、`-L$CUDA_HOME/lib64 -lcudart`
   が `ld: cannot find -lcudart` で落ちる。
3. **`$CUDA_HOME/lib64/stubs/libcuda.so` をノードの実ドライバへ向ける。**
   flashinfer は `-lcuda` でドライバにもリンクする。

`$SG_VENV/bin` を PATH に入れるのも必須 (`ninja` が subprocess で呼ばれる)。

### venv の後始末 (起動ログのノイズ潰し)

こちらは起動には影響しないが、放置すると `logs/*.err` が読めなくなる。
`src/venvsetup.py` の `tidy()` として毎回走る (どちらも冪等なので既存 venv にも効く)。

1. **`torchcodec` を消す。**
   sglang の依存だがノードに FFmpeg (`libavutil.so.56`-`.so.60`) が無く、
   `torchcodec/libtorchcodec_core{4..8}.so` の `dlopen` が5世代とも失敗する。
   sglang は握りつぶすものの、traceback をそのまま WARNING に埋め込むため
   `mimo_audio` / `mimo_v2_asr` x tokenizer+TP4 の 5 プロセスで
   **1起動あたり約1600行 (124KB)** になる。audio/video 入力は使わないうえ、
   `srt/utils/video_decoder.py` と `srt/utils/common.py` が
   decord / soundfile+torchaudio へのフォールバックを持っているので、
   消しても挙動は変わらない (video backend は decord になる)。
   FFmpeg をノードに入れられれば本来はそちらが正しいが、root が要る。
2. **`sarashina2_vision.py` の import を直す** (`venvsetup.py` の `patch_sarashina()`)。
   sglang 0.5.18 のバグで `MultimodalDataItem` を
   `managers.mm_utils` から import しているが、実体は
   `managers.schedule_batch` にある (mm_utils が再エクスポートしているのは
   `MultimodalInputs` だけ)。他の VL モデル (qwen2_vl 等) は
   schedule_batch から import しているので、そちらに揃える。
   該当行が無ければ何もしないので、upstream が直したら自動的に no-op になる。

`QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING=1` も同じ理由で入れてある。
cloudflared (quic-go) が UDP 受信バッファを 7MiB に広げようとして
`net.core.rmem_max` (208KiB) に阻まれる警告で、sysctl は root でないと変えられない。
実害は QUIC のスループット上限だけ。

cloudflared の
`Group ID 2200 is not between ping group 1 to 0` /
`ICMP proxy feature is disabled` は消せない。
`/proc/sys/net/ipv4/ping_group_range` が `1 0` (空) なので
非 root では ICMP ソケットを作れないというだけで、
ICMP プロキシは WARP からの ping 応答用。HTTP のトンネルには関係しない。

### コンテナを使っていない

`#SBATCH --container=` 付きのジョブは 2026-08-13 のサイト側コンテナ基盤更新以降、
計算ノードから `gpumng` への ssh (`docker inspect`) が
`Host key verification failed` で失敗して即死する
(`flux.job` の job 14018-14022 も 8/19 に同じ失敗)。
`~/.ssh/known_hosts` に `gpumng` のホスト鍵が無いのが直接の原因:

```bash
cp -a ~/.ssh/known_hosts ~/.ssh/known_hosts.bak.$(date +%Y%m%d)
for f in /etc/ssh/ssh_host_{ed25519,rsa,ecdsa}_key.pub; do
  awk '{print "gpumng,gpumng.cle.it-chiba.ac.jp,10.246.10.30 " $1 " " $2}' "$f"
done >> ~/.ssh/known_hosts
```

素のノードは Ubuntu 24.04 / glibc 2.39 / Python 3.12.3 / gcc 13.3 で
sglang 0.5.18 の wheel (manylinux_2_34, cp312) がそのまま入るため、
コンテナ無しで問題なく動く。戻す場合は `sglang.shinnarashino` の
`##SBATCH --container=` を有効化し、`rm -rf .venv` する
(イメージ側は Python 3.11 なので venv を作り直す必要がある)。

## クラスタ側の制約

『利用者向け操作マニュアル v12』5-1 / 5-3 (新習志野) より。

- research: 同時2ジョブ / **1ジョブ GPU5・CPU16・RAM480GB** / 24h / ノード跨ぎ不可
  → `#SBATCH` はこの範囲内 (GPU4 / CPU16 / 480G / 24h)
- ユーザ上限 GPU10 / CPU32 / RAM960GB
- **ストレージ 80GB (85GB で書込停止)**。docker image も含む。`hpcs_check_storage` で確認
  → venv 8.9GB + model 20GB で約29GB。docker image 32GB を消せば大きく空く
- 2026年8月から fairshare 導入。直近で使ったユーザは優先度が下がる
- 演習モード中は research ジョブが強制終了 → requeue (出力ファイルは上書きされる)
- コンテナ/ジョブはホストネットワーク共有。`SG_PORT` は他ユーザと衝突しうる。
  `FreeToken/.env` も 5050 なので同時起動不可

### home への大量 I/O について

マニュアル 7-3 に「ホームディレクトリはファイルサーバと計算サーバ間で常に同期を
とっています」「読み書きが頻発するようなジョブは実行しないで下さい」とある。
20GB の checkpoint を home から読むのは初回ロードの一度きり (実測 71s) なので
現状は許容範囲だが、本来は `/home/local/data/share_data/` に置いて
`/data` として読むのが正しい。書き込めるのは教員アカウントのみなので、
情報システム担当 (i-staff@chibatech.ac.jp / 内線0227) に配置を依頼し
`SG_MODEL=/data/Qwen3.8-27B-AWQ-INT4` にするのが本来の形。

## 障害時

エラーは「OOM」で括らず分類する。

| 症状 | 分類 | 対処 |
|---|---|---|
| `ld: cannot find -lcudart` / `-lcuda` | JIT リンク | `$CUDA_HOME/lib64` symlink farm が作られているか |
| `CUDA compiler and CUDA toolkit headers are incompatible` | JIT ツールチェイン | nvcc が 13.0.88 に固定されているか |
| `No such file or directory: 'ninja'` | JIT | PATH に `$SG_VENV/bin` |
| `Host key verification failed` | サイト container wrapper | 上記 known_hosts |
| ロード中に `torch.OutOfMemoryError` | weight VRAM OOM | `SG_MEM_FRACTION_STATIC` を下げる |
| `KV Cache is allocated` 付近で OOM | KV pool OOM | `SG_MEM_FRACTION_STATIC` / `SG_MAMBA_FULL_MEMORY_RATIO` |
| `Can not alloc mamba cache` | GDN state pool OOM | `SG_MAMBA_FULL_MEMORY_RATIO` を上げる。HiCache 併用時は upstream #36770 |
| CUDA graph capture で OOM | CUDA graph OOM | `SG_CUDA_GRAPH_MAX_BS` を下げる / `PYTORCH_CUDA_ALLOC_CONF` を外す |
| リクエスト処理中に OOM | prefill activation OOM | `SG_CHUNKED_PREFILL_SIZE` 2048→1024 |
| traceback 無しで kill、`oom-kill` | host RAM / cgroup OOM | `SG_HICACHE_SIZE` (rank単位!) と `SG_WEIGHT_LOAD_THREADS` を下げる |
| `Bus error` (SIGBUS) | checkpoint 消失 | home 同期の影響。`/data` へ移す |
| NCCL タイムアウト | TP 問題 | `/dev/shm` の空きを確認 |
| 画像リクエストで OOM (`materialize_multimodal_features`) | vision 動的確保 | vision encoder は static pool 外。`SG_MEM_FRACTION_STATIC` を下げる。`SG_IMAGE_PROCESSOR_BACKEND=pil` で前処理の cuda:0 使用 (実測1.4GiB) も止める |
| `Init torch distributed begin` の後で無音 | GPU 個体の P2P 故障 | `SG_NCCL_P2P_DISABLE=1`。`src/p2pdiag.py` で個体特定 |

`sacct` は当てにならない (サイトのラッパーが終了コードを Slurm に伝えないため、
失敗しても `COMPLETED 0:0` になることがある)。ログを見る。

```bash
grep -aoE "Load weight end[^\\]{0,60}|KV Cache is allocated[^\\]{0,80}|max_total_num_tokens[=: ][0-9]+|fired up and ready" logs/<jobid>.err
grep -aoE "Decode batch[^\\]{0,150}" logs/<jobid>.err | tail
srun --jobid=<jobid> --overlap nvidia-smi
```

**`logs/*.err` には `--api-key` が平文で入る** (sglang が server_args を丸ごと
INFO ログに出すため)。`logs/` は .gitignore 済みだが共有はしないこと。

## 既知の問題

- **分散初期化のハング (原因特定済 / 回避策あり / サイト側は未修理)。**
  TP=4 のジョブが一定確率で `Init torch distributed begin` →
  `sglang is using nccl==2.29.7` を出したまま無限に停止する。

  **原因は GPU 個体の P2P 故障。** 特定の物理 GPU に対して
  P2P (CUDA IPC / PCIe peer copy) で書き込むと、エラーも例外も出さずに
  **ゼロが返る**。`cudaDeviceCanAccessPeer` は全ペアで Y を返すので、
  NCCL はそれを信じて `Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM` の
  ring を張り、そのまま `all_reduce` が永久に返らない。

  ```
  canAccessPeer matrix (src -> dst):   actual peer copy:
    0: - Y Y Y                           0 -> 3: WRONG DATA sum=0.0
    1: Y - Y Y                           1 -> 3: WRONG DATA sum=0.0
    2: Y Y - Y                           2 -> 3: WRONG DATA sum=0.0
    3: Y Y Y -                           それ以外: ok
  ```

  ノードは GPU を10枚持っていてジョブは4枚しか掴まないので、
  **故障個体を引いたときだけ**ハングする。これが「一定確率で」「ノードを
  変えても起きる」「16時までは4回連続で成功していた」の正体。
  NCCL や SGLang の問題ではないし、`--dist-init-addr` や
  `--dist-timeout` では直らない (NCCL は待っているだけなので
  watchdog も発火しない)。

  切り分け用ツール:

  ```bash
  # NCCL を介さない単一プロセスの peer copy テスト。故障個体の UUID/PCI を出す
  srun --partition=research --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gpus-per-task=4 --mem=64G --time=00:10:00 src/p2pdiag.py

  # NCCL all_reduce だけの最小再現
  srun ... .venv/bin/torchrun --nproc_per_node=4 src/dist_smoke.py

  # 効く環境変数の総当たり
  srun ... src/dist_smoke_sweep.py
  ```

  総当たりの結果 (2026-08-30, gpu03):

  | 条件 | all_reduce |
  |---|---|
  | baseline | ハング |
  | `NCCL_CUMEM_ENABLE=0` | ハング |
  | **`NCCL_P2P_DISABLE=1`** | **成功** |
  | `NCCL_SHM_DISABLE=1` | ハング |
  | `NCCL_P2P_DISABLE=1` + `NCCL_CUMEM_ENABLE=0` | 成功 |
  | `NCCL_P2P_LEVEL=SYS` | ハング |

  **回避策**: `SG_NCCL_P2P_DISABLE=1` (既定)。`NCCL_P2P_DISABLE=1` を出し、
  同時に `--disable-custom-all-reduce` を付ける
  (custom all-reduce は NCCL を通さず自前で CUDA IPC を張るため、
  `NCCL_P2P_DISABLE` だけでは避けられない)。
  NCCL は共有メモリ経由 (ホスト中継) に落ちるので TP の帯域は落ちるが、
  A4500 は NVLink が無く P2P も PCIe 経由なので差は大きくない。

  **黙ってゼロが返る**以上、故障個体を掴んだまま「起動できてしまった」場合は
  TP の出力が壊れる可能性がある。P2P を切っておくのは速度以前に正しさの問題。

  サイト側の修理が必要なので情報システム担当
  (i-staff@chibatech.ac.jp / 内線0227) へ報告すること。
  `src/p2pdiag.py` の出力 (故障 GPU の UUID / PCI バス ID が入る) を添える。

  NCCL のログは **stdout** に出る。sglang のログは stderr なので、
  `NCCL_DEBUG=INFO` の出力は `logs/<jobid>.err` ではなく
  `logs/<jobid>.log` 側にある (「NCCL のログが1行も出ない」の正体)。

  起動ウォッチドッグ (`src/serve.py`, `SG_STARTUP_TIMEOUT` / `SG_STARTUP_RETRIES`) は
  そのまま残してある。故障個体を引いても張り直しで別の4枚を掴める可能性がある。
- **zombie request** (upstream sgl-project/sglang#36333 / #36876, 0.5.18 未修正)。
  クライアントが切断してもスケジューラに abort が届かず、`max_tokens` まで
  デコードし続けて running slot を占有する。`max_running_requests=4` だと
  1本で 25% を失う。**CodeAgent 側で `max_tokens` を必ず指定すること。**
  無指定だと context 一杯 (最大262k) まで回り続ける。
- HiCache + mamba の slot 枯渇 assert (#36770, 未修正)。本環境の
  50並列テストでは再現しなかったが、負荷が上がると出る可能性がある。
- `qwen3_coder` + thinking の token-0 ループ (#36537)。Qwen3.8-Flash-Next /
  SM120・SM121 の QSA sparse decode 固有で、本環境 (27B / SM86 / triton /
  speculative 無効) では再現しなかった。
- 起動時の `libtorchcodec` / `sarashina2_vision` の import エラーは
  2026-08-30 に潰した (「venv の後始末」参照)。どちらも無害だったが、
  前者が `logs/*.err` の 95% を占めていた。

## SGLang の更新

```bash
sed -i 's/^SG_PACKAGE=.*/SG_PACKAGE=sglang==<新版>/' .env
rm -rf .venv          # 次の投入で作り直される (nvcc pin も再適用される)
sbatch sglang.shinnarashino
```
