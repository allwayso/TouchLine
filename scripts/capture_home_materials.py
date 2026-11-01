"""从本地素材库截取首页使用的 WebP 预览图。"""
import asyncio
import argparse
from io import BytesIO
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

async def main():
    """读取指定服务的素材列表，将截图按实际 WebP 格式保存。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=Path('outputs/design-reference/materials.webp'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width':1440,'height':960},device_scale_factor=1)
        await page.goto(args.base_url.rstrip('/') + '/materials', wait_until='networkidle')
        await page.wait_for_timeout(2500)
        screenshot = await page.screenshot(type='png')
        with Image.open(BytesIO(screenshot)) as image:
            image.save(args.output, format='WEBP', quality=90)
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
