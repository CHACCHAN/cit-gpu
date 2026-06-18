"""
FLUX.1 Discord Bot
メンション + プロンプトで画像生成するシンプルBot。

機能:
  - @Bot <プロンプト> で画像生成をリクエスト
  - /health で空きパイプライン数を確認
  - 空きなし → 混雑中メッセージ（embed）
  - 空きあり → キューに追加して順次処理（embed）
"""
import os
import io
import asyncio
import aiohttp
import base64
from datetime import datetime

import discord
from discord.ext import commands

# ---- 設定 ----
DISCORD_TOKEN  = os.environ.get("DISCORD_TOKEN", "")
FLUX_API_URL   = os.environ.get("FLUX_API_URL",  "http://localhost:9090")
FLUX_API_KEY   = os.environ.get("FLUX_API_KEY",  "")
IMAGE_SIZE     = os.environ.get("IMAGE_SIZE",     "1024x1024")
IMAGE_STEPS    = int(os.environ.get("STEPS",      "50"))
IMAGE_GUIDANCE = float(os.environ.get("GUIDANCE", "3.5"))

# ---- Discord Bot 設定 ----
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ---- リクエストキュー ----
_request_queue: asyncio.Queue = asyncio.Queue()


def _api_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if FLUX_API_KEY:
        headers["Authorization"] = f"Bearer {FLUX_API_KEY}"
    return headers


def _embed_queued(prompt: str, position: int, avail: int, total: int) -> discord.Embed:
    embed = discord.Embed(
        title="📋 キューに追加しました",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="プロンプト", value=f"```{prompt[:200]}```", inline=False)
    embed.add_field(name="待機順",     value=f"{position} 番目",       inline=True)
    embed.add_field(name="パイプライン", value=f"{avail}/{total} 空き", inline=True)
    embed.set_footer(text="FLUX.1-dev • 生成完了までしばらくお待ちください")
    return embed


def _embed_generating(prompt: str, queue_remaining: int) -> discord.Embed:
    embed = discord.Embed(
        title="⏳ 生成中...",
        color=discord.Color.yellow(),
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="プロンプト", value=f"```{prompt[:200]}```", inline=False)
    if queue_remaining > 0:
        embed.add_field(name="後続待機", value=f"{queue_remaining} 件", inline=True)
    embed.set_footer(text="FLUX.1-dev • 50ステップ生成中")
    return embed


def _embed_done(prompt: str) -> discord.Embed:
    embed = discord.Embed(
        title="✅ 生成完了",
        color=discord.Color.green(),
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="プロンプト", value=f"```{prompt[:200]}```", inline=False)
    embed.set_footer(text="FLUX.1-dev")
    return embed


def _embed_busy(total: int) -> discord.Embed:
    embed = discord.Embed(
        title="🔴 ただいま混雑中",
        description=f"すべてのパイプライン（{total}本）がビジーです。\nしばらくしてから再度お試しください。",
        color=discord.Color.red(),
        timestamp=datetime.utcnow(),
    )
    embed.set_footer(text="FLUX.1-dev")
    return embed


def _embed_error() -> discord.Embed:
    embed = discord.Embed(
        title="❌ エラー",
        description="画像生成中にエラーが発生しました。しばらくしてから再試行してください。",
        color=discord.Color.dark_red(),
        timestamp=datetime.utcnow(),
    )
    embed.set_footer(text="FLUX.1-dev")
    return embed


async def check_availability() -> tuple[bool, int, int]:
    """空きパイプライン数を確認。Returns: (available, avail_count, total_count)"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{FLUX_API_URL}/health",
                headers=_api_headers(),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    avail = data.get("available_pipelines", 0)
                    total = data.get("total_pipelines",     0)
                    return avail > 0, avail, total
    except Exception as e:
        print(f"[bot] health check failed: {e}", flush=True)
    return False, 0, 0


async def generate_image(prompt: str) -> bytes | None:
    """FLUX APIに画像生成リクエストを送信してPNGバイト列を返す"""
    payload = {
        "prompt":          prompt,
        "n":               1,
        "size":            IMAGE_SIZE,
        "response_format": "b64_json",
        "steps":           IMAGE_STEPS,
        "guidance":        IMAGE_GUIDANCE,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{FLUX_API_URL}/v1/images/generations",
                json=payload,
                headers=_api_headers(),
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    b64 = data["data"][0]["b64_json"]
                    return base64.b64decode(b64)
                else:
                    print(f"[bot] API error: {resp.status} {await resp.text()}", flush=True)
    except Exception as e:
        print(f"[bot] generate failed: {e}", flush=True)
    return None


async def queue_worker():
    """キューからリクエストを取り出して順次処理するワーカー（無限ループ）"""
    print("[bot] Queue worker started.", flush=True)
    while True:
        try:
            message, prompt = await _request_queue.get()
            try:
                queue_remaining = _request_queue.qsize()

                # 「生成中」embedをチャンネルに送信（リプライではなくチャンネル宛）
                status_msg = await message.channel.send(
                    embed=_embed_generating(prompt, queue_remaining)
                )

                print(f"[bot] Generating: '{prompt[:60]}'", flush=True)
                img_bytes = await generate_image(prompt)

                if img_bytes:
                    file = discord.File(io.BytesIO(img_bytes), filename="flux_output.png")
                    embed = _embed_done(prompt)
                    embed.set_image(url="attachment://flux_output.png")
                    # 生成中メッセージを削除して画像付き完了メッセージを新規送信
                    # （editでは画像添付できないためこの方式）
                    await status_msg.delete()
                    await message.channel.send(embed=embed, file=file)
                else:
                    await status_msg.edit(embed=_embed_error())

            except Exception as e:
                print(f"[bot] Worker error: {e}", flush=True)
                try:
                    await message.channel.send(embed=_embed_error())
                except Exception:
                    pass
            finally:
                _request_queue.task_done()

        except Exception as e:
            print(f"[bot] Queue error: {e}", flush=True)
            await asyncio.sleep(1)


@bot.event
async def on_ready():
    print(f"[bot] Logged in as {bot.user} (ID: {bot.user.id})", flush=True)
    print(f"[bot] FLUX API: {FLUX_API_URL}", flush=True)
    asyncio.ensure_future(queue_worker())


@bot.event
async def on_message(message: discord.Message):
    # 自分自身・他Botのメッセージは無視
    if message.author.bot:
        return

    # メンションされていない場合は無視
    if bot.user not in message.mentions:
        return

    # プロンプト抽出（メンション部分を除去）
    prompt = message.content
    for mention in [f"<@{bot.user.id}>", f"<@!{bot.user.id}>"]:
        prompt = prompt.replace(mention, "")
    prompt = prompt.strip()

    if not prompt:
        embed = discord.Embed(
            title="💬 プロンプトを入力してください",
            description=f"`@{bot.user.display_name} a beautiful sunset over mountains`",
            color=discord.Color.blurple(),
        )
        await message.channel.send(embed=embed)
        return

    # 空き確認
    available, avail_count, total_count = await check_availability()

    if not available:
        await message.channel.send(embed=_embed_busy(total_count))
        return

    # キューに追加（受付アナウンスはチャンネル宛）
    position = _request_queue.qsize() + 1
    await _request_queue.put((message, prompt))
    await message.channel.send(embed=_embed_queued(prompt, position, avail_count, total_count))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("[bot] ERROR: DISCORD_TOKEN is not set.", flush=True)
        exit(1)
    bot.run(DISCORD_TOKEN)