# Dataset: GuessMovie
The dataset can be downloaded from the [link](https://drive.google.com/file/d/1CQot9g5dYSm2xG7Q-n0CoWpTGXi6w0vG). You should decompress it and move all the files to the current directory.

- `guessmovie_doc.json`: 16,881 documents with structured KB.
- `guessmovie_dialog.json`: 13,434 simulated dialogues.
- `wo_entity`: preprocessed documents for MD3 model.
  - `guessmovie_dialog_doc_id.json`: the part for sumulating dialogue.
  - `doc_rep`: the part for training documents representation.
  - `vocab`: all 16881 documents with word id and vocabulary.

> All documents are split on two parts. The former is for training Doc-Rep and NLU. And the latter is for training dialogue policy and simulating dialogue.
