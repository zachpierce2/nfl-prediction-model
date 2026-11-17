"""Expected-starter challenger using existing QB passing formula and paired grading."""
import os
from pathlib import Path
import qb_expected
import qb_passing

ROOT=Path('data/auto_market/expected_qb_shadow')
VERSION='market_epa_expected_qb_passing_v1'

def record(schedule_path,core_rows,folder):
    if not core_rows:return dict(status='NO_NEW_CORE_FORECASTS',new_qb_forecasts=0)
    path=os.environ.get('QB_EXPECTATIONS_PATH')
    if not path:return dict(status='EXPECTED_QB_COLLECTION_UNAVAILABLE',new_qb_forecasts=0)
    return qb_passing.record(schedule_path,core_rows,folder,reports_path=path,
        resolver=qb_expected.resolve,root=ROOT,version=VERSION,
        id_fields=['home_expected_qb_id','away_expected_qb_id'],tag='expected_qb')

def grade_shadow(grader,core_rows,finals,quotes,now):
    return qb_passing.grade_shadow(grader,core_rows,finals,quotes,now,root=ROOT,version=VERSION)
