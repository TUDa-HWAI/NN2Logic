import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from nn2logic import QHybridCreator

QHybridCreator("qat-w-samples.json")