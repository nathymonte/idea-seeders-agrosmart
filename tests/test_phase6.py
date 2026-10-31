import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_sensor_pipeline import TEST_CONFIG, sensor_event, trusted_event, write_csv

from scripts.demo import build_demo
from services.dashboard_data import image_frame
from services.data_lake import (
    load_image_analysis_summary,
    parse_datetime,
    read_dataset,
    recompute_field_status,
    save_trusted_sensor_reading,
    write_json,
)
from services.sensor_validator import validate_sensor_event
from services.threshold_config import ThresholdConfigError, get_threshold_config
from src.prepare_data import prepare_rows
from src.streaming.consumer import consume_messages


class Phase6Test(unittest.TestCase):
    def test_malformed_identifiers_do_not_interrupt_next_message(self):
        invalid = sensor_event('bad')
        invalid['event_id'] = {'unexpected': 'object'}
        invalid['field_id'] = 123
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            consumed = consume_messages([SimpleNamespace(value=invalid), SimpleNamespace(value=sensor_event('good'))],
                                        base_path=tmp, threshold_config=TEST_CONFIG, output_csv=Path(tmp) / 'absent.csv')
            self.assertEqual(consumed, 2)
            self.assertEqual(len(read_dataset('trusted', 'sensor_readings', tmp)), 1)
            self.assertEqual(len(read_dataset('rejected', 'sensor_readings', tmp)), 1)

    def test_partition_dates_use_utc(self):
        timestamp = parse_datetime('2026-08-24T23:30:00-03:00')
        self.assertEqual(timestamp.isoformat(), '2026-08-25T02:30:00+00:00')

    def test_incremental_refinement_preserves_other_fields_and_ignores_old_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trusted_sensor_reading(trusted_event(event_id='old', timestamp='2026-07-01T08:00:00Z'), tmp)
            save_trusted_sensor_reading(trusted_event(event_id='current'), tmp)
            other = Path(tmp) / 'refined/field_status/TALHAO-02.json'
            write_json(other, {'untouched': True})
            old = next((Path(tmp) / 'trusted').rglob('month=07/day=01/*.jsonl'))
            old.write_text('corrupt old partition')
            recompute_field_status(tmp, Path(tmp) / 'missing.csv', TEST_CONFIG,
                                   field_reference_times={'TALHAO-01': parse_datetime('2026-08-24T08:00:00Z')})
            self.assertEqual(json.loads(other.read_text()), {'untouched': True})
            status = json.loads((Path(tmp) / 'refined/field_status/TALHAO-01.json').read_text())
            self.assertEqual(status['sensor_summary']['readings_count'], 1)

    def test_late_event_does_not_regress_latest_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trusted_sensor_reading(trusted_event(event_id='recent', timestamp='2026-08-24T18:00:00Z'), tmp)
            recompute_field_status(tmp, Path(tmp) / 'missing.csv', TEST_CONFIG)
            save_trusted_sensor_reading(trusted_event(event_id='late', timestamp='2026-08-24T08:00:00Z'), tmp)
            recompute_field_status(tmp, Path(tmp) / 'missing.csv', TEST_CONFIG,
                                   field_reference_times={'TALHAO-01': parse_datetime('2026-08-24T08:00:00Z')})
            status = json.loads((Path(tmp) / 'refined/field_status/TALHAO-01.json').read_text())
            self.assertEqual(status['latest_reading']['event_id'], 'recent')

    def test_nonfinite_sensor_values_are_rejected(self):
        for value in [float('nan'), float('inf'), float('-inf'), 'NaN', 'Infinity']:
            with self.subTest(value=value):
                normalized, errors = validate_sensor_event(sensor_event('invalid', soil=value), threshold_config=TEST_CONFIG)
                self.assertIsNone(normalized)
                self.assertTrue(errors)

    def test_missing_or_unsafe_field_is_rejected(self):
        for field in [None, '', '../escape', 'A/B']:
            event = sensor_event('invalid')
            if field is None:
                del event['field_id']
            else:
                event['field_id'] = field
            self.assertIsNone(validate_sensor_event(event, threshold_config=TEST_CONFIG)[0])

    def test_images_are_scoped_by_field_and_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_csv = Path(tmp) / 'images.csv'
            rows = [{'field_id': 'TALHAO-01', 'data_analise': '2026-08-24T07:00:00Z', 'predicted_label': 'Sick'},
                    {'field_id': 'TALHAO-02', 'data_analise': '2026-08-24T07:00:00Z', 'predicted_label': 'Health'},
                    {'field_id': 'TALHAO-02', 'data_analise': '2026-08-01T07:00:00Z', 'predicted_label': 'Sick'}]
            write_csv(image_csv, rows)
            for field in ['TALHAO-01', 'TALHAO-02']:
                save_trusted_sensor_reading(trusted_event(event_id=field, field_id=field), tmp)
            recompute_field_status(tmp, image_csv, TEST_CONFIG)
            statuses = {p.stem: json.loads(p.read_text()) for p in (Path(tmp) / 'refined/field_status').glob('*.json')}
            self.assertEqual(statuses['TALHAO-01']['attention_level'], 'medium')
            self.assertEqual(statuses['TALHAO-02']['attention_level'], 'normal')
            self.assertEqual(statuses['TALHAO-02']['latest_image_analysis']['total_images'], 1)

    def test_legacy_unassigned_images_do_not_raise_field_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'legacy.csv'
            write_csv(csv_path, [{'data_analise': '2026-08-24', 'predicted_label': 'Sick', 'localidade': 'Talhao A'}])
            self.assertIsNone(load_image_analysis_summary(csv_path, 'TALHAO-01'))
            frame = image_frame(csv_path)
            self.assertEqual(frame.iloc[0]['metadata_source'], 'legacy_simulated')
            self.assertEqual(frame.iloc[0]['field_id'], '')

    def test_last_reading_risk_is_not_hidden_by_healthy_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            for hour in range(8, 18):
                save_trusted_sensor_reading(trusted_event(event_id=str(hour), timestamp=f'2026-08-24T{hour:02}:00:00Z'), tmp)
            save_trusted_sensor_reading(trusted_event(event_id='risk', timestamp='2026-08-24T18:00:00Z', soil=20), tmp)
            recompute_field_status(tmp, Path(tmp) / 'absent.csv', TEST_CONFIG)
            result = json.loads((Path(tmp) / 'refined/field_status/TALHAO-01.json').read_text())
            self.assertGreater(result['sensor_summary']['average_soil_moisture_percent'], 40)
            self.assertEqual(result['attention_level'], 'medium')
            self.assertEqual(result['latest_reading']['event_id'], 'risk')
            self.assertTrue(result['decision_reasons'])

    def test_bounded_query_skips_old_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trusted_sensor_reading(trusted_event(event_id='old', timestamp='2026-07-01T08:00:00Z'), tmp)
            save_trusted_sensor_reading(trusted_event(event_id='recent'), tmp)
            old_path = next((Path(tmp) / 'trusted').rglob('month=07/day=01/*.jsonl'))
            old_path.write_text('corrupt old partition')
            start = datetime(2026, 8, 24, tzinfo=UTC)
            records = read_dataset('trusted', 'sensor_readings', tmp, start=start)
            self.assertEqual([r['event_id'] for r in records], ['recent'])

    def test_idle_poll_flushes_before_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = iter([0, 0, 6, 6, 6])
            order = []
            with patch('src.streaming.consumer.recompute_field_status', side_effect=lambda *a, **k: order.append('refine')):
                with redirect_stdout(io.StringIO()):
                    consume_messages([SimpleNamespace(value=sensor_event('one')), None], base_path=tmp,
                                     threshold_config=TEST_CONFIG, clock=lambda: next(ticks), refine_seconds=5,
                                     acknowledge=lambda m: order.append('ack'))
            self.assertEqual(order, ['refine', 'ack'])

    def test_failed_refinement_does_not_acknowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            acknowledged = []
            with patch('src.streaming.consumer.recompute_field_status', side_effect=OSError('disk')):
                with self.assertRaises(OSError), redirect_stdout(io.StringIO()):
                    consume_messages([SimpleNamespace(value=sensor_event('one'))], base_path=tmp,
                                     threshold_config=TEST_CONFIG, refine_every=1, acknowledge=acknowledged.append)
            self.assertEqual(acknowledged, [])

    def test_keyboard_interrupt_flushes_pending_batch(self):
        def messages():
            yield SimpleNamespace(value=sensor_event('one'))
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as tmp:
            with patch('src.streaming.consumer.recompute_field_status') as refine:
                with self.assertRaises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
                    consume_messages(messages(), base_path=tmp, threshold_config=TEST_CONFIG)
            refine.assert_called_once()

    def test_atomic_json_failure_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            write_json(path, {'old': True})
            with self.assertRaises(ValueError):
                write_json(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'old': True})
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_prepare_rows_uses_explicit_metadata_without_diagnosis(self):
        data = {'meta': {'generated_at': '2026-08-24T08:00:00Z'},
                'results': [{'image_name': 'unknown.jpg', 'predicted_label': 'Sick', 'confidence': .8,
                             'scores': {'Health': .2, 'Sick': .8}}]}
        row = prepare_rows(data)[0]
        self.assertEqual(row['field_id'], '')
        self.assertEqual(row['is_correct'], '')
        self.assertEqual(row['anomalia'], 'Nao diagnosticada')
        associated = prepare_rows(data, {'unknown.jpg': {'field_id': 'TALHAO-02', 'actual_label': 'Sick'}})[0]
        self.assertTrue(associated['is_correct'])
        self.assertEqual(associated['label_source'], 'manifest')

    def test_invalid_yaml_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.yaml'
            path.write_text('sensor_fields: [')
            with self.assertRaises(ThresholdConfigError):
                get_threshold_config(path)

    def test_complete_demo_preserves_layers_and_three_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                summary = build_demo(Path(tmp))
            self.assertEqual((summary['accepted'], summary['rejected']), (72, 1))
            self.assertEqual(len(read_dataset('raw', 'sensors', Path(tmp) / 'data_lake')), 73)
            statuses = {p.stem: json.loads(p.read_text())['attention_level']
                        for p in (Path(tmp) / 'data_lake/refined/field_status').glob('*.json')}
            self.assertEqual(statuses, {'TALHAO-01': 'normal', 'TALHAO-02': 'medium', 'TALHAO-03': 'high'})
            with self.assertRaises(ValueError):
                build_demo(Path(tmp))
