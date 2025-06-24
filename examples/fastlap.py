# Add the parent directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Now import fastlap
import fastlap
print("Available functions:", [f for f in dir(fastlap) if not f.startswith('_')])
print("Can access solve_lap?", hasattr(fastlap, 'solve_lap'))