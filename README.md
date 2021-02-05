# Multi-Document Driven Dialogue (MD3)
This is the code for AAAI2021 paper [Converse, Focus and Guess - Towards Multi-Document Driven Dialogue](https://arxiv.org/abs/2102.02435).

## Requirements
- Ubuntu 16.04
- Python >= 3.6.0
- PyTorch >= 1.3.0

## Dataset: GuessMovie
We build a benchmark GuessMovie dataset for MD3 task on the base of the dataset WikiMovies (Miller et al. 2016). It includes 16,881 documents with 6 different attributes (i.e. directed_by, release_year, written_by, starred_actors, has_genre, in_language). The dataset can be downloaded from the [link](https://mega.nz/file/AKogESCC#P-30oCiN8yUeq9vGAbVpctbcVjoj1IVh6iA9BfLs8ZU), and should be decompressed to the data directory of this repository.

<img src="https://github.com/laddie132/MD3/raw/main/imgs/example.png" width="500" alt="" align=center/>

## Preprocess
To preprocess, we provide several methods.

```bash
python preprocess.py --[vocab/split]
```

- `vocab`: splitting words, making vocabulary, word to id and filtering GloVe embeddings.
- `split`: splitting the dataset to two parts. The former is for training Doc-Rep and NLU. And the latter is for training dialogue policy and simulating dialogue.

## Training Doc-Rep
To obtain attribute-aware document representation, just run the following command.
```bash
./train_doc_rep.sh
```

> You can modify the bash script to change directory of input and output.

## Training NLU
We use imitation learning to train NLU module separately on the former part of dataset.
```bash
python train_nlu.py --out [OUT_INFIX] --train --test
```

## Training Policy
We use reinforce learning to train Policy module on the latter part of dataset.
```bash
python run_game.py --in [IN_INFIX] --out [OUT_INFIX] --train --test
```

- `IN_INFIX`: directory name of NLU checkpoints
- `OUT_INFIX`: directory name of output checkpoints for NLU and Policy

## Testing Game
Testing the dialog with 5k simulations on the latter part of dataset.
```bash
python run_game.py --in [IN_INFIX] --test
```

- `IN_INFIX`: directory name of NLU and Policy checkpoints

> You can change the `agent_type` in `config/game_config.yaml` to test different agents.

## Others
Some tools are in `tests` directory.

## Reference
If you consider our work useful, please cite the paper:
```
@inproceedings{liu2021converse,
  title={Converse, Focus and Guess - Towards Multi-Document Driven Dialogue},
  author={Liu, Han and Yuan, Caixia and Wang, Xiaojie and Yang, Yushu and Jiang, Huixing and Wang, Zhongyuan},
  booktitle={Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021}
}
```