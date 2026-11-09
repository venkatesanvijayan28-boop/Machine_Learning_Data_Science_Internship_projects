"""
Master Orchestration Build Script
Runs the model training pipeline followed by visualization and notebook generation.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_models import main as train_main
from src.generate_visualizations_and_notebook import main as viz_main

def main():
    print("=" * 80)
    print("STARTING CALIFORNIA HOUSING END-TO-END BUILD PROCESS")
    print("=" * 80)
    
    print("\n>>> STEP 1: TRAINING REGRESSION MODELS & COMPUTING METRICS")
    train_main()
    
    print("\n>>> STEP 2: GENERATING PLOTS & EXECUTED JUPYTER NOTEBOOK")
    viz_main()
    
    print("\n" + "=" * 80)
    print("BUILD PROCESS COMPLETE: ALL ARTIFACTS, MODELS, AND NOTEBOOK GENERATED")
    print("=" * 80)

if __name__ == '__main__':
    main()
