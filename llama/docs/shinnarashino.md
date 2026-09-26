# 新習志野 Bonsai

RTX A4500 20GB ×5 の1ジョブで Ternary Bonsai 2 27B (Qwen3.8-27B の ternary 版) を
262144 tokens ×8 slot で提供する。モデルIDは `bonsai-2-27b-pq2_0`。**実機では未検証。**

| 項目 | 値 | 理由 |
|---|---|---|
| モデル | `prism-ml/Ternary-Bonsai-2-27B-gguf` revision `b072e1d3b35a0a630cece372c2127528e0994386` の PQ2_0 (7.2GB) | Ampere では PTQ1_0 より速い |
| llama.cpp | `PrismML-Eng/llama.cpp` commit `adfffbe41b2cabcd51fff326ab045662265062bb` (branch `prism`) | PQ2_0 は fork でしか読めない |
| 分割 | `-sm layer` | row / tensor 分割は PQ2_0 で読めない |
| slot | `-c 2097152 --parallel 8 --no-kv-unified` | 各 slot 262144 (学習時の上限) |
| KV | q8_0 | 16 attention 層 × KV head 4 × 256 次元 = 34 KiB/token → ×262144×8 = 68 GiB。重み 6.7 GiB と合わせて約 97 GiB に収まる想定 |

## 初回セットアップ

新習志野の home は津田沼と別なので、このフォルダを git で持っていき、向こうで実行する。

```bash
cd ~/cit-gpu/llama && mkdir -p logs
# nvcc (CUDA 12.x) が PATH にあること。無ければ津田沼で実行して llama.cpp-prism/ を rsync する
python3 src/build.py --repo https://github.com/PrismML-Eng/llama.cpp \
    --ref adfffbe41b2cabcd51fff326ab045662265062bb --arch 86 --dest llama.cpp-prism
python3 src/fetch.py --repo prism-ml/Ternary-Bonsai-2-27B-gguf \
    --revision b072e1d3b35a0a630cece372c2127528e0994386 \
    --dest models/Bonsai-2-27B Ternary-Bonsai-2-27B-PQ2_0.gguf
mkdir -p -m 700 .secrets
(umask 077; echo '<API key>' > .secrets/api-keys)
bash bonsai.shinnarashino --dry-run
sbatch bonsai.shinnarashino
```

Tunnel は張らない設定。公開するときは token を `.secrets/tunnel-token.shinnarashino` に置き、
ジョブの `launch.py` 引数に `--tunnel-token .secrets/tunnel-token.shinnarashino` を足す
(cloudflared は `../SGLang/bin/cloudflared` を使う)。

## 検証

```bash
NODE=$(squeue -h -j <jobid> -o %N)
python3 src/verify.py --url http://$NODE:5050 --slots 8 --body '{"reasoning_effort":"none"}'
# 8 slot 同時に約25万 tokens: 末尾に --all-long
srun --jobid=<jobid> --overlap nvidia-smi   # GPU ごとの使用量
```

## 注意

- P2P が黙ってゼロを返す GPU がある (`../SGLang/README.md`)。ジョブで
  `GGML_CUDA_P2P` を外し、GPU 間転送をホスト経由にしている
- 推論モデルで既定の reasoning effort は `xhigh`。`max_tokens` が小さいと本文が空になる。
  16384 以上にするか `reasoning_effort: "medium"`。`"high"` は HTTP 500、思考を切るのは `"none"`
- system message は先頭に1つだけ (それ以外は HTTP 500)
- 約200K tokens の量子化 KV で接続リセットの報告がある。出たら `-ctk f16 -ctv f16` で slot を減らす
- port 5050 は SGLang / FreeToken の新習志野ジョブと同じ。同一ノードでは衝突する
- ストレージ上限 80GB (モデル 7.2GB + ビルド約0.5GB)
