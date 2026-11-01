"""检查展示首页的桌面、手机、减少动态效果与素材库入口。"""
import asyncio
import argparse
import json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    """检查指定本地服务并保存截图与结构化结果；失败时返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8766')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/design-reference/home-review'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width':1440,'height':1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('response', lambda response: errors.append(f'{response.status} {response.url}') if response.status >= 400 else None)
        await page.goto(args.base_url.rstrip('/') + '/', wait_until='networkidle')
        await page.evaluate('document.fonts.ready')
        await page.screenshot(path=str(out/'desktop-hero.png'))
        await page.screenshot(path=str(out/'desktop-full.png'), full_page=True)
        desktop = await page.evaluate("""() => ({overflow:document.documentElement.scrollWidth>innerWidth,title:document.querySelector('h1').innerText,fonts:document.fonts.check('700 20px Barlow')})""")
        await page.locator('#capture').scroll_into_view_if_needed()
        await page.wait_for_timeout(1200)
        await page.screenshot(path=str(out/'capture.png'))
        await page.set_viewport_size({'width':390,'height':844})
        await page.evaluate('scrollTo(0,0)')
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(out/'mobile-hero.png'))
        await page.screenshot(path=str(out/'mobile-full.png'),full_page=True)
        mobile = await page.evaluate('({overflow:document.documentElement.scrollWidth>innerWidth})')
        await page.emulate_media(reduced_motion='reduce')
        await page.wait_for_timeout(300)
        reduced = await page.evaluate("""() => ({stopped:[...document.querySelectorAll('.film-row')].every(row=>getComputedStyle(row).animationName==='none' || getComputedStyle(row).animationPlayState==='paused'),animation:getComputedStyle(document.querySelector('.film-row')).animationName})""")
        images = await page.evaluate("""async () => {
            const images = [...document.images];
            images.forEach(image => image.loading = 'eager');
            await Promise.all(images.map(image => image.decode().catch(() => {})));
            return images.filter(image => !image.naturalWidth).map(image => image.src);
        }""")
        await page.get_by_role('link',name='Open Workspace',exact=True).first.click()
        await page.wait_for_url('**/materials')
        result = {'desktop':desktop,'mobile':mobile,'reducedMotion':reduced,'brokenImages':images,'cta':page.url,'errors':errors}
        (out/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        await browser.close()
        assert not desktop['overflow'] and not mobile['overflow'], '页面出现横向溢出'
        assert desktop['fonts'], '本地字体加载失败'
        assert reduced['stopped'], '减少动态效果设置未停止动画'
        assert not images and not errors, '页面存在资源或脚本错误'

if __name__ == '__main__':
    asyncio.run(main())
