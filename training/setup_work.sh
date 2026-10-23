#!/bin/bash
# Creates the work directory the training scripts run in and puts the
# released split there under the names the scripts read (data/f10/{aclass,bclass}_qa_{train,test}.json).
#   E=${TRIPHYS_WORK:-<repo>/work}
#   $E/scripts      -> training/scripts        (python scripts/..., bash scripts/...)
#   $E/fc/training  -> training                (fc/training/n1/..., fc/training/teacher_pilot/...)
# Usage: bash training/setup_work.sh
set -e
REPO=$(dirname "$(dirname "$(readlink -f "$0")")")
E=${TRIPHYS_WORK:-$REPO/work}
mkdir -p $E/data/f10 $E/data/p1 $E/data/p2 $E/data/teacher_full $E/results $E/logs $E/outputs/phase1 $E/outputs/phase2 $E/fc
ln -sfn $REPO/training/scripts $E/scripts
ln -sfn $REPO/training $E/fc/training
cp $REPO/data/mpr_train.json $E/data/f10/aclass_qa_train.json
cp $REPO/data/mpr_test.json $E/data/f10/aclass_qa_test.json
cp $REPO/data/kpr_train.json $E/data/f10/bclass_qa_train.json
cp $REPO/data/kpr_test.json $E/data/f10/bclass_qa_test.json
echo "work dir $E ready"
