#!/usr/bin/env bash

# change config on 'dataset/doc_id_path', 'global/cand_doc_num', 'train/train_iters' and 'train/test_iters'

export DEVICE_ID=0
export OUT_DIR=doc-rep-32-5k

# train & test
python train_doc_rep.py --slot=directed_by --train --test  --out=$OUT_DIR --gpuid $DEVICE_ID
python train_doc_rep.py --slot=release_year --train --test --out=$OUT_DIR --gpuid $DEVICE_ID
python train_doc_rep.py --slot=written_by --train --test --out=$OUT_DIR --gpuid $DEVICE_ID
python train_doc_rep.py --slot=starred_actors --train --test --out=$OUT_DIR --gpuid $DEVICE_ID
python train_doc_rep.py --slot=has_genre --train --test --out=$OUT_DIR --gpuid $DEVICE_ID
python train_doc_rep.py --slot=in_language --train --test --out=$OUT_DIR --gpuid $DEVICE_ID

# build representation
python build_doc_rep.py --slot=directed_by --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID
python build_doc_rep.py --slot=release_year --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID
python build_doc_rep.py --slot=written_by --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID
python build_doc_rep.py --slot=starred_actors --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID
python build_doc_rep.py --slot=has_genre --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID
python build_doc_rep.py --slot=in_language --in=$OUT_DIR --out=$OUT_DIR --gpuid $DEVICE_ID