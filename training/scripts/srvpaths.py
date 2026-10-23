"""Paths used by the training code in training/n1, training/scripts and training/teacher_pilot.

  E            work directory: env TRIPHYS_WORK, default <repo>/work (created by training/setup_work.sh)
  SCRIPTS      training/scripts
  TRAINING     training/
  REPO         repository root
  N1           training/n1 (native-tool SFT training and evaluation)
  EVAL_LIB     training/evallib (prompts, response parser, metrics)
  BASE_MODEL   env BASE_MODEL, default Qwen/Qwen3.5-4B
  EGOEXO_ROOT  env EGOEXO_ROOT, MPR videos <EGOEXO_ROOT>/<take_uid>/best_exo.mp4
  BAAI_ROOT    env BAAI_ROOT, KPR videos <BAAI_ROOT>/<folder_id>/<video_id>.mp4
"""
import os

SCRIPTS = os.path.dirname(os.path.realpath(__file__))
TRAINING = os.path.dirname(SCRIPTS)
REPO = os.path.dirname(TRAINING)
N1 = os.path.join(TRAINING, 'n1')
EVAL_LIB = os.path.join(TRAINING, 'evallib')
E = os.environ.get('TRIPHYS_WORK', os.path.join(REPO, 'work'))
BASE_MODEL = os.environ.get('BASE_MODEL', 'Qwen/Qwen3.5-4B')
EGOEXO_ROOT = os.environ.get('EGOEXO_ROOT', '').rstrip('/')
BAAI_ROOT = os.environ.get('BAAI_ROOT', '').rstrip('/')
