# 新習志野 Bonsai

RTX A4500 20GB ×5 の1ジョブで Ternary Bonsai 2 27B (Qwen3.8-27B の ternary 版) を提供する。
GPU 1 枚に llama-server を 1 つ (262144 tokens ×1 slot) 載せて 5 つ並べ、
Caddy (`bin/caddy`、設定は `common/Caddyfile`) が port 5050 で振り分ける。
同じクライアント (Hermes は会話ごとに 1 つ) は cookie で同じ GPU に固定し、prompt cache を再利用する。
公開URLは `https://gpgpu.cc-chacchan.com/v1`、モデルIDは `bonsai-2-27b-pq2_0`。

| 項目 | 値 | 理由 |
|---|---|---|
| モデル | `prism-ml/Ternary-Bonsai-2-27B-gguf` revision `b072e1d3b35a0a630cece372c2127528e0994386` の PQ2_0 (7.2GB) | Ampere では PTQ1_0 より速い |
| llama.cpp | `PrismML-Eng/llama.cpp` commit `adfffbe41b2cabcd51fff326ab045662265062bb` (branch `prism`) | PQ2_0 は fork でしか読めない |
| 配置 | `--replicas 5`、各 `-c 262144 --parallel 1` | 5 枚で layer 分割するより速く、slot も多い (下記) |
| KV | q8_0 | 16 attention 層 × KV head 4 × 256 次元 = 34 KiB/token → 262144 で 8.5 GiB |
| prompt cache | `--cache-ram 32768` ×5 (host RAM 計 160 GiB) | 1 GPU 1 slot なので、押し出された会話は RAM から戻す。`--cache-reuse` はこのモデル (recurrent 層あり) では無効になる |
| chat template | `shinnarashino/bonsai.jinja` | 元の GGUF のものは reasoning_effort が xhigh/medium/low 以外だと HTTP 500。minimal→low、high/max/その他→xhigh に読み替える |

## 初回セットアップ

新習志野の home は津田沼と別なので、このフォルダを git で持っていき、向こうで実行する。

```bash
cd ~/cit-gpu/llama && mkdir -p logs
# CUDA 12.2 は既定の gcc 13 を受け付けないので g++-11 を host compiler にする
CUDAHOSTCXX=g++-11 python3 common/build.py --repo https://github.com/PrismML-Eng/llama.cpp \
    --ref adfffbe41b2cabcd51fff326ab045662265062bb --arch 86 --dest llama.cpp-prism
python3 common/fetch.py --repo prism-ml/Ternary-Bonsai-2-27B-gguf \
    --revision b072e1d3b35a0a630cece372c2127528e0994386 \
    --dest models/Bonsai-2-27B Ternary-Bonsai-2-27B-PQ2_0.gguf
# bin/caddy を置く (手順は ../README.md)
cp .env.example .env && chmod 600 .env   # LLAMA_API_KEY と TUNNEL_TOKEN を書く
bash shinnarashino/bonsai --dry-run
sbatch shinnarashino/bonsai
```

## 検証

```bash
NODE=$(squeue -h -j <jobid> -o %N)   # /slots は Cloudflare 経由では 403 なので LAN から
python3 common/verify.py --url http://$NODE:5050 --slots 5 --replicas 5 --body '{"reasoning_effort":"none"}'
# 5 slot 同時に約25万 tokens: 末尾に --all-long
srun --jobid=<jobid> --overlap nvidia-smi   # GPU ごとの使用量
```

## 構成の比較 (実測)

| 構成 | 最も重い GPU | 生成速度 |
|---|---|---|
| 1 枚 ×5 (採用) | 17.0 GB | 1 本 約 45 tok/s、5 並列 合計 約 170 tok/s |
| 5 枚 layer 分割、4 slot × 262144 | 15.6 GB | 1 本 約 43 tok/s、4 並列 合計 約 75 tok/s |
| 5 枚 layer 分割、8 slot × 262144 | OOM | — |

layer 分割では q8_0 KV が flash attention の前に f16 へ展開され、全 slot 合計の context に比例した
compute buffer (合計 2097152 tokens で約 8.5 GiB/GPU) を各 GPU が持つ。attention 16 層が 5 枚に
均等に割れず 1 枚が 4 層を持つこともあり、8 slot × 262144 は収まらない。
1 枚 ×1 slot なら重み 6.5 + KV 8.5 + compute 1.4 GiB で収まる。

## 注意

- 推論モデルで既定の reasoning effort は `xhigh`。`max_tokens` が小さいと本文が空になる。
  16384 以上にするか `reasoning_effort: "medium"`。思考を切るのは `"none"`
- system message は先頭に1つだけ (それ以外は HTTP 500)
- 約200K tokens の量子化 KV で接続リセットの報告がある
- 新しいクライアントはその時点で処理中が最も少ない replica に割り当てる。固定先が処理中なら空いている別の replica へ送り
  (prompt cache は効かない)、5 台とも処理中なら最大 45 秒空きを待ち、空かなければ 503 + `Retry-After: 2`。llama-server の中で順番待ちになった要求は
  stream でも始まるまで 1 byte も返らず (実測 136 秒無通信)、Cloudflare の 100 秒で 524 になるため
- stream は prompt 読み込み中に 30 秒ごとに `:` が届くので、長い prompt でも Cloudflare の 100 秒で切れない
- `/slots` `/props` は振り分け先 1 台分しか返さない。verify.py は `--replicas 5` で 1 台あたりの slot 数を確かめる
- port 5050-5055 を使う。5050 は SGLang / FreeToken の新習志野ジョブと同じなので同一ノードでは衝突する
- ストレージ上限 80GB (モデル 7.2GB + ビルド約0.5GB)
