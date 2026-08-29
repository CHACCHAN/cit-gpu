# Qwen3.8-27B を SGLang で配信 (新習志野 / gpumng)

新習志野 GPGPU 1号機 (`gpumng.cle.it-chiba.ac.jp`, RTX A4500 20GB x4, TP=4) で
SGLang を動かし、OpenAI 互換 API を CodeAgent 向けに提供する。
津田沼 (`gpumng2`, H100 NVL x2) の DeepSeek-V4-Flash 構成からの移行。

`sglang.shinnarashino` 1ファイルで完結。設定は `.env` のみ。

## 使い方

```bash
cd SGLang
cp .env.example .env      # SG_API_KEY と TUNNEL_TOKEN を入れる
sbatch sglang.shinnarashino
tail -f logs/<jobid>.err  # sglang のログは stderr
```

初回は venv 作成 (約8分 / 8.9GB) とモデル取得 (20GB) が走る。
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

`sglang.shinnarashino` が venv 作成後にやっている3点。どれも必須。

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

- **分散初期化のハング (最優先 / 未解決)。**
  TP=4 のジョブが一定確率で `Init torch distributed begin` →
  `sglang is using nccl==2.29.7` を出したまま無限に停止する。
  `--dist-timeout` を短くしてもタイムアウトせず、`NCCL_DEBUG=INFO` でも
  NCCL のログが1行も出ないので、NCCL に入る手前で止まっている。
  `--dist-init-addr` を明示しても変わらない。
  **SGLang 固有ではない**: sglang を介さない素の 4GPU torchrun all_reduce でも
  同じハングが再現する (job 14175 / 14190 が hang、14176 / 14191 は同条件で成功)。
  ノード固有でもなく gpu02 / gpu03 の両方で起きる。
  16:00 頃までは 4 回連続で正常起動していた (job 14164/14169/14170/14171)。

  再現用の最小ジョブ:

  ```bash
  #SBATCH --partition=research --gpus-per-task=4 ...
  cat > ar.py <<'PY'
  import torch, torch.distributed as dist
  dist.init_process_group("nccl")
  r = dist.get_rank(); torch.cuda.set_device(r)
  t = torch.ones(1024, device=f"cuda:{r}")
  dist.all_reduce(t); print(f"rank {r} OK", flush=True)
  PY
  timeout 90 ./.venv/bin/torchrun --nproc_per_node=4 ar.py
  ```

  暫定対応として `sglang.shinnarashino` に起動ウォッチドッグを入れてある
  (`SG_STARTUP_TIMEOUT` 秒で READY にならなければ殺して張り直す、最大
  `SG_STARTUP_RETRIES` 回)。恒久対応は情報システム担当への報告
  (i-staff@chibatech.ac.jp / 内線0227) が必要。上の最小再現ジョブを添えること。
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
- 起動時の `libtorchcodec` / `sarashina2_vision` の import エラーは無害
  (sglang が握りつぶす)。

## SGLang の更新

```bash
sed -i 's/^SG_PACKAGE=.*/SG_PACKAGE=sglang==<新版>/' .env
rm -rf .venv          # 次の投入で作り直される (nvcc pin も再適用される)
sbatch sglang.shinnarashino
```
