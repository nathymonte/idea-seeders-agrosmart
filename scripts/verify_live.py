import argparse
import json
from types import SimpleNamespace

from playwright.sync_api import sync_playwright

from services.data_lake import read_dataset, write_json
from services.paths import PROJECT_ROOT
from src.streaming.consumer import consume_messages
from src.streaming.producer import build_event


def verify(url: str, browser_path: str):
    lake = PROJECT_ROOT / '.demo/data_lake'
    image_csv = PROJECT_ROOT / '.demo/output/classificacoes.csv'
    before = len(read_dataset('trusted', 'sensor_readings', lake))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_path, headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.goto(url)
        page.get_by_text(f'{before} leituras no periodo', exact=False).wait_for(timeout=30000)
        consume_messages([SimpleNamespace(value=build_event('TALHAO-01', 'normal'))],
                         base_path=lake, output_csv=image_csv, refine_every=1)
        page.get_by_text(f'{before + 1} leituras no periodo', exact=False).wait_for(timeout=20000)
        page.get_by_role('combobox').first.click()
        page.get_by_role('option', name='TALHAO-01', exact=True).click()
        page.get_by_text('Umidade do solo', exact=True).first.wait_for()
        page.screenshot(path=str(PROJECT_ROOT / 'artifacts/dashboard/desktop-talhao.png'))
        browser.close()
    result = {'automatic_refresh': True, 'before': before, 'after': before + 1, 'field_filter': True}
    write_json(PROJECT_ROOT / 'artifacts/dashboard/live-check.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verifica atualizacao da demonstracao adicionando uma leitura valida.')
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--browser', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.browser), indent=2))
