import urllib.request, os, stat

urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', './cloudflared')
os.chmod('./cloudflared', os.stat('./cloudflared').st_mode | stat.S_IEXEC)