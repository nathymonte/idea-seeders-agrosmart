"""Browser smoke checks for the running dashboard; screenshots are local artifacts."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def verify(url: str, browser_executable: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable, headless=True)
        for name, size in [('desktop', {'width': 1440, 'height': 1000}),
                           ('mobile', {'width': 390, 'height': 844})]:
            page = browser.new_page(viewport=size, device_scale_factor=1)
            page.goto(url)
            page.get_by_role('tab', name='Ambiente', exact=True).wait_for(timeout=30000)
            page.locator('.js-plotly-plot').first.wait_for(timeout=30000)
            assert page.get_by_text('TALHAO-01', exact=False).count() > 0
            assert page.locator('[data-testid="stException"]').count() == 0
            page.screenshot(path=str(destination / f'{name}-ambiente.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
            page.locator('.js-plotly-plot').first.scroll_into_view_if_needed()
            page.screenshot(path=str(destination / f'{name}-evolucao.png'))
            page.get_by_role('tab', name='Folhas', exact=True).click()
            page.locator('[data-testid="stImage"] img').first.wait_for()
            image = page.locator('[data-testid="stImage"] img').first
            assert image.evaluate('(img) => img.complete && img.naturalWidth > 0')
            page.screenshot(path=str(destination / f'{name}-folhas.png'), full_page=True)
            page.get_by_role('tab', name='Dados', exact=True).click()
            with page.expect_download() as download:
                page.get_by_role('button', name='Exportar leituras').click()
            exported = download.value
            exported.save_as(str(destination / f'{name}-leituras.csv'))
            assert (destination / f'{name}-leituras.csv').stat().st_size > 100
            results.append({'viewport': name, 'charts': True, 'image': True, 'export': True,
                            'horizontal_overflow': False})
            page.close()
        browser.close()
    (destination / 'browser-checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--browser', required=True)
    parser.add_argument('--out', type=Path, default=Path('artifacts/dashboard'))
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.browser, args.out), indent=2))
