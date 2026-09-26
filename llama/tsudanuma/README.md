# 津田沼 MiMo

H100 NVL ×2 で MiMo V2.6 Flash RL Q3_K と Heretic LoRA を提供する。
公開URLは `https://gpgpu2.cc-chacchan.com/v1`、モデルIDは `mimo-v2.6-flash-rl-q3_k`。

| 項目 | 値 |
|---|---|
| llama.cpp | `ggml-org/llama.cpp` commit `a25c9865fe03c954c93fd755b5d79ae86ba99750` (sm_90) |
| モデル | `AesSedai/MiMo-V2.6-Flash-RL-GGUF` revision `05c13439c18ba7183cb6afe294924c75ba7aa7b4` の Q3_K 4 shard (148GB) |
| LoRA | `MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF` revision `0ecc6864d187d4117d3348c85379bfe9298f49e3` |
| KV / slot | q8_0、524288 tokens ×3 (unified pool)。理論値で GPU0 約 85 / 93.5 GiB (下記) |

## 初回セットアップ

ストレージ上限 240GB のうちモデルで約150GB を使う。

```bash
cd /home/share/L24G3102/cit-gpu/llama && mkdir -p logs
python3 common/build.py --repo https://github.com/ggml-org/llama.cpp \
    --ref a25c9865fe03c954c93fd755b5d79ae86ba99750 --arch 90 --dest llama.cpp
for i in 1 2 3 4; do
    python3 common/fetch.py --repo AesSedai/MiMo-V2.6-Flash-RL-GGUF \
        --revision 05c13439c18ba7183cb6afe294924c75ba7aa7b4 \
        --dest models Q3_K/MiMo-V2.6-Flash-RL-Q3_K-0000$i-of-00004.gguf
done
python3 common/fetch.py --repo MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF \
    --revision 0ecc6864d187d4117d3348c85379bfe9298f49e3 \
    --dest adapters MiMo-V2.6-Flash-RL-Uncensored-Heretic-lora.gguf
cp .env.example .env && chmod 600 .env   # LLAMA_API_KEY と TUNNEL_TOKEN を書く
bash tsudanuma/mimo --dry-run
```

nvcc はログインノードの `/usr/local/cuda-12.4/bin` にある (PATH に無ければ足す)。
cloudflared は `../SGLang/bin/cloudflared` を使う。

## 運用

```bash
sbatch tsudanuma/mimo
tail -f logs/<jobid>.log logs/<jobid>.err
```

停止は `scancel <jobid>`。24時間制限があるので、続けて動かすなら
`sbatch --dependency=singleton tsudanuma/mimo` で次を予約しておく。
LoRA を外すときは `tsudanuma/mimo` の `--lora` 行を消して再投入する。

## 検証

```bash
python3 common/verify.py --url https://gpgpu2.cc-chacchan.com --slots 3 --slot-ctx 524288   # 長文は --long-context / --all-long
python3 tsudanuma/verify_lora.py --url https://gpgpu2.cc-chacchan.com
```

## VRAM の見積もり (512K ×3、理論値・実機未検証)

全体 attention は 48 層中 9 層 (0,5,11,17,23 が GPU0、29,35,41,47 が GPU1)。
KV は 4 head × (192+128) 次元 × q8_0 = 1,360 B/token/層。残り 39 層は 128 tokens の sliding window で無視できる。

| GPU0 (重い方) | 262144 ×4 | 524288 ×3 |
|---|---|---|
| 重み (Q3_K 137.8 GiB の半分) | 68.9 | 68.9 |
| KV (5 層) | 6.6 | 10.0 |
| q8_0 KV の f16 展開 (合計 tokens × 2,560 B、最悪値) | 2.5 | 3.8 |
| その他 compute / CUDA context (仮定) | 2.0 | 2.0 |
| 合計 GiB (93.5 GiB 中) | 約 80 | 約 85 |

524288 ×4 は約 89.5 GiB で余裕がない。
