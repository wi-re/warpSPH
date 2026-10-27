"""Long-run output times must not stall or step past the paper snapshots."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_blobPaperLong import next_dt


@pytest.mark.parametrize('target',[.25,1.,4.,8.])
def test_timestep_lands_on_output_without_zero_next_step(target):
    kh=2.9030989544096926;end=8*kh;t=target*kh-1.e-4
    dt=next_dt(.0005,t,end,kh)
    assert t+dt==pytest.approx(target*kh,abs=1.e-14)
    assert next_dt(.0005,t+dt,end,kh)==.0005


def test_output_schedule_keeps_a_smaller_cfl_step():
    assert next_dt(1.e-5,.1,8.,1.)==1.e-5
