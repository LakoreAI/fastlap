try:
    from ._fastlap_rs import solve_lap
except ImportError:
    from _fastlap_rs import solve_lap

def supported_algos():
    return ["lapjv", "hungarian", "lapmod"]

__all__ = ['solve_lap', 'supported_algos']