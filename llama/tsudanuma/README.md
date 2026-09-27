# 津田沼 MiMo

H100 NVL ×2 で MiMo V2.6 Flash RL Q3_K と Heretic LoRA を配信する。

- API: `https://gpgpu2.cc-chacchan.com/v1`
- モデルID: `mimo-v2.6-flash-rl-q3_k`
- 構成: 256K tokens (`262144`) × 6 並列、KV は q8_0

## 初回セットアップ

`llama/` で実行する。Gateway の導入は [共通 README](../README.md) を参照。

```bash
mkdir -p logs
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
cp .env.example .env && chmod 600 .env
# .env に LLAMA_API_KEY と TUNNEL_TOKEN を設定
```

## 起動・確認

```bash
sbatch tsudanuma/mimo
srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 \
    python3 common/verify.py --url http://127.0.0.1:5051 --slots 6 --slot-ctx 262144
srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 \
    python3 tsudanuma/verify_lora.py --url http://127.0.0.1:5050
```

`common/verify.py` は worker の `/props` と `/slots` を使うため、worker の 5051 を指定する。
公開 API は [共通 README](../README.md) の SGLang Gateway 契約に従う。

ログは `tail -f logs/<jobid>.log logs/<jobid>.err`、停止は `scancel <jobid>`。
