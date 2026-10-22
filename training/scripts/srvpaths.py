"""Machine paths of the training code. The scripts in training/n1, training/scripts and training/teacher_pilot are the
ones run on the training server; only their absolute paths were replaced by the names below.

  server path                                      here
  /root/triworld/experiment_paper                  E = $TRIPHYS_WORK (default <repo>/work; training/setup_work.sh)
  /root/triworld/experiment_paper/scripts          SCRIPTS = training/scripts
  /root/triworld/experiment_paper/fc/training/n1   N1 = training/n1
  /root/triworld/experiment_right                  EVAL_LIB = training/evallib (prompts, parser, metrics)
  /root/triworld/models/Qwen3.5-4B                 BASE_MODEL = $BASE_MODEL (default Qwen/Qwen3.5-4B)
  /root/triworld/egoexo4d_selected/takes           EGOEXO_ROOT = $EGOEXO_ROOT  (<take_uid>/best_exo.mp4)
  /root/triworld/BAAI                              BAAI_ROOT = $BAAI_ROOT      (<folder_id>/<video_id>.mp4)
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
