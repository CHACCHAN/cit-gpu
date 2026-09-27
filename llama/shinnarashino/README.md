# 新習志野 Bonsai

RTX A4500 ×5 で Ternary Bonsai 2 27B PQ2_0 を配信する。

- API: `https://gpgpu.cc-chacchan.com/v1`
- モデルID: `bonsai-2-27b-pq2_0`
- 構成: GPU 1枚につき 160K tokens (`163840`) × 2枠、合計10並列。KV は q8_0
- `/v1` は SGLang Model Gateway の `cache_aware` で5台に振り分ける
- 公開 API は [共通 README](../README.md) の SGLang Gateway 契約に従う
- 160K ×2 は 2026-09-27 に A4500 実機で起動と同時2件を確認済み (19,504 MiB 使用)。168K ×2 以上は VRAM 不足で起動不可

## 初回セットアップ

リポジトリを新習志野側に配置し、`llama/` で実行する。Gateway の導入は [共通 README](../README.md) を参照。

```bash
mkdir -p logs
CUDAHOSTCXX=g++-11 python3 common/build.py --repo https://github.com/PrismML-Eng/llama.cpp \
    --ref adfffbe41b2cabcd51fff326ab045662265062bb --arch 86 --dest llama.cpp-prism
python3 common/fetch.py --repo prism-ml/Ternary-Bonsai-2-27B-gguf \
    --revision b072e1d3b35a0a630cece372c2127528e0994386 \
    --dest models/Bonsai-2-27B Ternary-Bonsai-2-27B-PQ2_0.gguf
cp .env.example .env && chmod 600 .env
# .env に LLAMA_API_KEY と TUNNEL_TOKEN を設定
```

## 起動・確認

```bash
sbatch shinnarashino/bonsai
for port in 5051 5052 5053 5054 5055; do
    srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 \
        python3 common/verify.py --url http://127.0.0.1:$port \
        --slots 2 --slot-ctx 163840 --body '{"reasoning_effort":"none"}'
done
```

`common/verify.py` は worker の `/props` と `/slots` を使うため、各 worker のポートを指定する。

ログは `tail -f logs/<jobid>.log logs/<jobid>.err`、停止は `scancel <jobid>`。
