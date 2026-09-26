# 津田沼 MiMo

H100 NVL ×2 で MiMo V2.6 Flash RL Q3_K と Heretic LoRA を提供する。
公開URLは `https://gpgpu2.cc-chacchan.com/v1`、モデルIDは `mimo-v2.6-flash-rl-q3_k`。

| 項目 | 値 |
|---|---|
| llama.cpp | `ggml-org/llama.cpp` commit `a25c9865fe03c954c93fd755b5d79ae86ba99750` (sm_90) |
| モデル | `AesSedai/MiMo-V2.6-Flash-RL-GGUF` revision `05c13439c18ba7183cb6afe294924c75ba7aa7b4` の Q3_K 4 shard (148GB) |
| LoRA | `MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF` revision `0ecc6864d187d4117d3348c85379bfe9298f49e3` |
| KV / slot | q8_0、262144 tokens ×4 (unified pool) |

## 初回セットアップ

ストレージ上限 240GB のうちモデルで約150GB を使う。

```bash
cd /home/share/L24G3102/cit-gpu/llama && mkdir -p logs
python3 src/build.py --repo https://github.com/ggml-org/llama.cpp \
    --ref a25c9865fe03c954c93fd755b5d79ae86ba99750 --arch 90 --dest llama.cpp
for i in 1 2 3 4; do
    python3 src/fetch.py --repo AesSedai/MiMo-V2.6-Flash-RL-GGUF \
        --revision 05c13439c18ba7183cb6afe294924c75ba7aa7b4 \
        --dest models Q3_K/MiMo-V2.6-Flash-RL-Q3_K-0000$i-of-00004.gguf
done
python3 src/fetch.py --repo MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF \
    --revision 0ecc6864d187d4117d3348c85379bfe9298f49e3 \
    --dest adapters MiMo-V2.6-Flash-RL-Uncensored-Heretic-lora.gguf
mkdir -p -m 700 .secrets
(umask 077; echo '<API key>' > .secrets/api-keys; echo '<Tunnel token>' > .secrets/tunnel-token)
bash mimo.tsudanuma --dry-run
```

nvcc はログインノードの `/usr/local/cuda-12.4/bin` にある (PATH に無ければ足す)。
cloudflared は `../SGLang/bin/cloudflared` を使う。

## 運用

```bash
sbatch mimo.tsudanuma
tail -f logs/<jobid>.log logs/<jobid>.err
```

停止は `scancel <jobid>`。24時間制限があるので、続けて動かすなら
`sbatch --dependency=singleton mimo.tsudanuma` で次を予約しておく。
LoRA を外すときは `mimo.tsudanuma` の `--lora` 行を消して再投入する。

## 検証

```bash
python3 src/verify.py --url https://gpgpu2.cc-chacchan.com   # 長文は --long-context / --all-long
python3 src/verify_lora.py --url https://gpgpu2.cc-chacchan.com
```
