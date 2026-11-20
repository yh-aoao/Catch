"""Run with python -m catch_then_basket.test_handoff (no renderer)."""
from .environment import StableHold


def main():
    hold=StableHold(seconds=.3,relative_speed=.3,palm_distance=.18)
    for _ in range(7):assert not hold.update(.04,True,.1,.1)
    assert hold.update(.04,True,.1,.1)
    assert not hold.update(.04,False,.1,.1)
    assert hold.elapsed==0.
    for invalid in [(True,.31,.1),(True,.1,.19),(True,float('nan'),.1)]:
        hold.elapsed=.28
        assert not hold.update(.04,*invalid)
        assert hold.elapsed==0.
    hold.elapsed=.28
    assert not hold.update(.04,True,.1,.1,safe=False)
    print('Stable hold checks passed')

if __name__=='__main__':main()
