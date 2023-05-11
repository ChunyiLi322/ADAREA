## Setup

- install python >= 3.6 and pip
- `pip install -r requirements.txt`
- install [PyTorch](https://pytorch.org)
- Download [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) (glove.840B.300d) to `resources/`

Data used in the paper are prepared as follows:

### SNLI

- Download and unzip [SNLI](https://www.dropbox.com/s/0r82spk628ksz70/SNLI.zip?dl=0) 
(pre-processed by [Tay et al.](https://github.com/vanzytay/EMNLP2018_NLI)) to `data/orig`. 
- Unzip all zip files in the "data/orig/SNLI" folder. (`cd data/orig/SNLI && gunzip *.gz`)
- `cd data && python prepare_snli.py` 

### SciTail

- Download and unzip [SciTail](http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip) 
dataset to `data/orig`.
- `cd data && python prepare_scitail.py`

### Quora

- Download and unzip [Quora](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)
dataset (pre-processed by [Wang et al.](https://github.com/zhiguowang/BiMPM)) to `data/orig`.
- `cd data && python prepare_quora.py`

### WikiQA

- Download and unzip [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419)
to `data/orig`.
- `cd data && python prepare_wikiqa.py`
- Download and unzip [evaluation scripts](http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz). 
Use the `make -B` command to compile the source files in `qg-emnlp07-data/eval/trec_eval-8.0`.
Move the binary file "trec_eval" to `resources/`.

## Usage

To train a new text matching model, run the following command: 

```bash
python train.py $config_file.json5
python train.py configs/main.json5
```

To check the configurations only, use

```bash
python train.py $config_file.json5 --dry
```

To evaluate an existed model, use `python evaluate.py $model_path $data_file`, here's an example:

```bash
python evaluate.py models/snli/benchmark/best.pt data/snli/train.txt 
python evaluate.py models/snli/benchmark/best.pt data/snli/test.txt 
```
