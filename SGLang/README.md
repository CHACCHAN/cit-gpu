# DeepSeek-V4-Flash-0731 を SGLang で配信

津田沼 GPGPU (`gpumng2`, H100 NVL 94GB x2) 上で SGLang を動かし、既存の
Cloudflare Tunnel 経由で OpenAI 互換 API を公開する。FreeToken 0.1.2 からの移行。

`sglang.job` 1ファイルで完結。設定は `.env` のみ。

## 使い方

```bash
cd SGLang
cp .env.example .env      # TUNNEL_TOKEN を入れる
sbatch sglang.job
tail -f logs/<jobid>.log
```

初回は venv 作成（約20分 / 9GB）とモデル取得（約167GB）が走る。
`The server is fired up and ready to roll!` が出れば稼働。停止は `scancel <jobid>`。

## API

```bash
curl -s http://localhost:5050/v1/models | python3 -m json.tool

curl -s http://localhost:5050/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],
       "max_tokens":512,"stream":false}' | python3 -m json.tool
```

Open WebUI: Base URL `https://gpgpu.cc-chacchan.com/v1` / API key は任意の非空文字 /
Model `deepseek-v4-flash`。reasoning は `reasoning_content` で返る。
このホスト名は Cloudflare Access 配下なので、未認証だと Tunnel が正常でも 403 になる。

## context を伸ばす

`.env` の `SG_CONTEXT_LENGTH` を上げて再投入するだけ。

```
65536 -> 131072 -> 262144 -> 393216 -> 524288 -> 786432 -> 1048576
```

`--context-length` は1リクエストの上限で、VRAM は予約しない。実際に載る量は
`gpu_mem * SG_MEM_FRACTION_STATIC - 重み` で決まる KV pool 側なので、起動できても
足りているとは限らない。長いプロンプトで確認する。ログの `max_total_num_tokens`
が実際の pool サイズ。

## 設定の要点

- **`SG_MOE_RUNNER_BACKEND=marlin` は変えない。** この checkpoint の routed expert は
  MXFP4（FP4 を int8 にパック + E8M0 block-32 scale, 約143GB）で、Hopper には FP4
  テンソルコアが無いため W4A16 でしか動かない。sglang 0.5.18 は marlin 指定時のみ
  `Mxfp4MarlinMoEMethod` を選び、既定の `auto` では FP8 として int8 パックを誤読する。
  FP8 版（`sgl-project/DeepSeek-V4-Flash-FP8`）は 294GB あり 188GiB には載らない。
- `--attention-backend` / `--page-size` / `--kv-cache-dtype` / `--quantization` は
  **渡していない**。sglang が `DeepseekV4ForCausalLM` から `dsv4` / `256` /
  `fp8_e4m3` / config.json 由来を自動で入れる。
- chat template は checkpoint に無い（DeepSeek V4 は Python エンコーダを使う）。
  sglang が内蔵しているので `/v1/chat/completions` はそのまま動く。
- FreeToken で必要だった `libnccl.so.2` の symlink 回避策は不要。venv の
  `nvidia-nccl-cu13` が torch 経由で使われる（実測 `nccl==2.29.7`）。

## クラスタ側の制約

『利用者向け操作マニュアル v12』(津田沼) より。`sacctmgr` の値より厳しい。

- 同時ジョブ 2件 / **GPU2・CPU16・256GB**（ユーザ単位もジョブ単位も）/ 24h
  → `#SBATCH` はこの上限どおりなので `--mem` はこれ以上上げられない
- ストレージ 240GB（docker image も含む）。`hpcs_check_storage` で確認
- コンテナはホストネットワーク共有。`SG_PORT` は他ユーザと衝突しうる
- 演習モード中は research ジョブが強制終了→再キューされる

### home は常時同期される（重要）

マニュアル 7-3 に「ホームディレクトリはファイルサーバと計算サーバ間で常に同期を
とっています」「読み書きが頻発するようなジョブは実行しないで下さい」とある。

156GiB を home から毎回読むのはこれに該当し、初回実行（job 13616）では
ロード中に checkpoint が消え、両 TP rank が SIGBUS で落ちた
（quota 使用量が 213GB→58.5GB とちょうど checkpoint 分だけ減少）。

大容量の重みは `/home/local/data/share_data/` に置くのが正しい。各ノードの
ローカル NVMe に複製され、ジョブ内では `/data` に read-only マウントされる。
ただし書き込めるのは教員アカウントのみなので、情報システム担当
(i-staff@chibatech.ac.jp / 内線0227) に配置を依頼し、`.env` を

```
SG_MODEL=/data/DeepSeek-V4-Flash-0731
```

にするのが本来の形。ロードも速くなる。

## 障害時

GPU OOM とホスト RAM OOM を区別する。

| 症状 | 対処 |
|---|---|
| ロード中に `Bus error` (SIGBUS) | checkpoint の存在を確認 → `/data` へ移す → `SG_WEIGHT_LOAD_THREADS` を下げる |
| ロード中に `torch.OutOfMemoryError` | `SG_MEM_FRACTION_STATIC` を下げる |
| `KV Cache is allocated` 付近で OOM | `SG_CONTEXT_LENGTH` を下げる |
| リクエスト処理中に OOM | `SG_CHUNKED_PREFILL_SIZE` 4096→2048 |
| traceback 無しで kill、`oom-kill` | ホスト RAM。`SG_WEIGHT_LOAD_THREADS` を下げる |
| `MXFP4 Marlin requires SM90+` | `SG_MOE_RUNNER_BACKEND=flashinfer_mxfp4` |
| ポート使用中 | `SG_PORT` を変えて Tunnel の ingress も合わせる |
| CUDA graph の捕獲で失敗 | `sglang.job` の `PYTORCH_CUDA_ALLOC_CONF` を外す |

`sacct` は当てにならない（サイトのコンテナラッパーが終了コードを Slurm に
伝えないため、失敗しても `COMPLETED 0:0` になる）。ログを見る。

```bash
grep -nE 'avail mem|Load weight end|max_total_num_tokens' logs/<jobid>.log
srun --jobid=<jobid> --overlap nvidia-smi
```

## SGLang の更新

```bash
sed -i 's/^SG_PACKAGE=.*/SG_PACKAGE=sglang==<新版>/' .env
rm -rf .venv          # 次の投入で作り直される
sbatch sglang.job
```
