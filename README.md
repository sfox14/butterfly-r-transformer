# Butterfly + R-Transformer
This repository combines R-Transformer with Butterfly matrices for faster and more efficient computation of Linear and RNN layers. The code depends on the CUDA implementation of [Kaleidoscope: An Efficient, Learnable Representation For All Structured Linear Maps](https://openreview.net/forum?id=BkgrBgSYDS) which is available from [here](https://github.com/HazyResearch/butterfly). It can be installed from our fork with:
```
git clone https://github.com/sfox14/butterfly.git
python setup.py install
python -c "from torch_butterfly.butterfly import Butterfly; print('SUCCESS')"
```

# R-Transformer
The Pytorch implementation of [R-Transformer](https://arxiv.org/abs/1907.05572) comes from [DSE/R-Transformer](https://github.com/DSE-MSU/R-transformer).  Some parts of the code are adapted from the implementation of [TCN](https://github.com/locuslab/TCN) and [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 


For more details about R-Transformer, Please read the [paper](https://arxiv.org/abs/1907.05572), and if you find this work useful and use it on your research, please cite:

```
@article{wang2019rtransf,
  title={R-Transformer: Recurrent Neural Network Enhanced Transformer},
  author={Wang, Zhiwei and Ma, Yao and Liu, Zitao and Tang, Jiliang},
  journal={arXiv preprint arXiv:1907.05572},
  year={2019}
}
```

## Usage
The repository is arranged as follows:
```
[Task Name] /
    data/ # contains the datasets
    experiment.py #run experiment 
    model.py # comtains the model
    utils.py # utility functions including dataset downloading and preprocessing
models /
    RTransformer.py # RTransformer model    
```
The dataset for the "polyphonic music modeling" and "language word modelling" experiments are already included in audio/data/ and language_word/data respectively. For other experiments that are based on much larger datasets, the data needs to be downloaded and put into the "data" folder which should be created firstly. 

To run the language word modelling task with Butterfly multiplications on the PTB dataset:

```
cd language_word/
mkdir output
python experiment.py --rnn_type=CUSTOM --butterfly
```
This will save a log file and model checkpoint to ./output directory.


