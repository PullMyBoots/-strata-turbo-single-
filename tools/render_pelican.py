"""Capture original model-written HTML with Chromium and export animated previews."""
import argparse
import base64
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--session', default='strata-readme')
    p.add_argument('--seconds', type=int, default=6)
    p.add_argument('--fps', type=int, default=20)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = a.input.absolute()
    destination = a.output.absolute()
    destination.mkdir(parents=True, exist_ok=True)
    frames_dir = root / 'output/playwright' / a.session
    frames_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Path.home() / '.codex/skills/playwright/scripts/playwright_cli.sh'
    cli = ['bash', str(wrapper)] if wrapper.exists() else ['npx', '--yes', '--package', '@playwright/cli', 'playwright-cli']
    cli += ['-s=' + a.session]
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(destination), **kwargs)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runs = json.loads((source / 'results.json').read_text())['runs']
    counts = {row['arm']: 0 for row in runs}
    checks = []
    subprocess.run(cli + ['open', f'http://127.0.0.1:{server.server_port}/'], cwd=root, check=True,
                   stdout=(frames_dir / 'open.txt').open('w'))
    try:
        for row in runs:
            counts[row['arm']] += 1
            name = f"{row['arm']}-{counts[row['arm']]}"
            folder = destination / name
            folder.mkdir(exist_ok=True)
            original = source / f"{row['index']}-{row['arm']}" / 'pelican.html'
            shutil.copy2(original, folder / 'pelican.html')
            url = f'http://127.0.0.1:{server.server_port}/{name}/pelican.html'
            subprocess.run(cli + ['tab-new', url], cwd=root, check=True,
                           stdout=(frames_dir / f'{name}-tab.txt').open('w'))
            subprocess.run(cli + ['goto', url], cwd=root, check=True,
                           stdout=(frames_dir / f'{name}-navigate.txt').open('w'))
            subprocess.run(cli + ['snapshot'], cwd=root, check=True,
                           stdout=(frames_dir / f'{name}-snapshot.txt').open('w'))
            code = '''async (page) => {
 await page.setViewportSize({width:1000,height:720});
 await page.clock.install({time:new Date('2026-10-10T00:00:00Z')});
 // The browser context can retain a clock from a previous tab; pause in its future.
 await page.clock.pauseAt(await page.evaluate(() => Date.now() + 60000));
 const external=[];
 await page.route('**/*',async route=>{
  const u=route.request().url();
  if(u.startsWith('BASE') || u.startsWith('data:') || u.startsWith('blob:')) await route.continue();
  else {external.push(u);await route.abort();}
 });
 await page.goto('URL',{waitUntil:'domcontentloaded'});
 await page.evaluate(()=>{
  for(const s of document.querySelectorAll('svg')) if(s.pauseAnimations)s.pauseAnimations();
  for(const a of document.getAnimations())a.pause();
 });
 const info=await page.evaluate(()=>({title:document.title,svgs:document.querySelectorAll('svg').length,
  smil:document.querySelectorAll('animate,animateTransform,animateMotion').length,
  css:document.getAnimations().length,scripts:document.querySelectorAll('script').length,
  scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight}));
 const cdp=await page.context().newCDPSession(page);
 const frames=[];
 for(let i=0;i<FRAMES;i++){
  if(i>0)await page.clock.runFor(1000/FPS);
  await page.evaluate(t=>{
   for(const s of document.querySelectorAll('svg'))if(s.setCurrentTime)s.setCurrentTime(t);
   for(const a of document.getAnimations()){a.pause();a.currentTime=t*1000;}
  },i/FPS);
  // CDP captures the manually controlled timeline without waiting for virtual RAF.
  frames.push((await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false})).data);
 }
 await page.unroute('**/*');
 await cdp.detach();
 await page.clock.resume();
 return {...info,external_requests:external,frames};
}'''.replace('BASE', f'http://127.0.0.1:{server.server_port}/').replace('URL', url).replace('FRAMES', str(a.seconds*a.fps)).replace('FPS', str(a.fps)).replace('NAME', name)
            r = subprocess.run(cli + ['run-code', code], cwd=root, text=True, capture_output=True, timeout=300)
            (frames_dir / f'{name}-render.txt').write_text(r.stdout + r.stderr)
            r.check_returncode()
            parsed = json.JSONDecoder().raw_decode(r.stdout.split('### Result', 1)[1].lstrip())[0]
            encoded_frames = parsed.pop('frames')
            for index, encoded in enumerate(encoded_frames):
                (frames_dir / f'{name}-{index:03}.png').write_bytes(base64.b64decode(encoded))
            # Save browser checks without duplicating all base64 PNGs.
            (frames_dir / f'{name}-render.txt').write_text(json.dumps(parsed, indent=2) + '\n')
            images = [Image.open(frames_dir / f'{name}-{i:03}.png').convert('RGB')
                      for i in range(a.seconds * a.fps)]
            hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
            palette = images[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
            frames[0].save(folder / 'pelican.gif', save_all=True, append_images=frames[1:],
                           duration=1000//a.fps, loop=0, optimize=False, disposal=2)
            images[0].save(folder / 'preview.png')
            contact = Image.new('RGB', (2000, 1440), 'white')
            for index, second in enumerate((0, 1, 2, 3)):
                contact.paste(images[second * a.fps], ((index % 2) * 1000, (index // 2) * 720))
            contact.save(frames_dir / f'{name}-contact.png')
            checks.append({'run': name, 'html_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                           'unique_frames': len(set(hashes)), 'animated': len(set(hashes)) > 10,
                           'seconds': a.seconds, 'fps': a.fps, 'browser_log': f'{name}-render.txt',
                           'gif_bytes': (folder / 'pelican.gif').stat().st_size, 'browser': parsed,
                           'self_contained': not parsed['external_requests']})
            print(name, checks[-1], flush=True)
            (destination / 'render-checks.json').write_text(json.dumps(checks, indent=2) + '\n')
            subprocess.run(cli + ['tab-close'], cwd=root, check=True, stdout=subprocess.DEVNULL)
    finally:
        server.shutdown()
        subprocess.run(cli + ['close'], cwd=root, stdout=subprocess.DEVNULL)
    (destination / 'render-checks.json').write_text(json.dumps(checks, indent=2) + '\n')


if __name__ == '__main__':
    main()
