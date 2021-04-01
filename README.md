# Paraphrastic Representations at Scale

Code to train models from "Paraphrastic Representations at Scale".

The code is written in Python 3.7 and requires H5py, jieba, numpy, scipy, sentencepiece, sacremoses, and PyTorch >= 1.0 libraries. These can be insalled with the following command:

    pip install -r requirements.txt

To get started, download the data files used for training from http://www.cs.cmu.edu/~jwieting and download the STS evaluation data:

    wget http://phontron.com/data/paraphrase-at-scale.zip
    unzip paraphrase-at-scale.zip
    rm paraphrase-at-scale.zip
    wget http://www.cs.cmu.edu/~jwieting/STS.zip .
    unzip STS.zip
    rm STS.zip
    
If you use our code, models, or data for your work please cite:

    @inproceedings{wieting19simple,
        title={Simple and Effective Paraphrastic Similarity from Parallel Translations},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url={https://arxiv.org/abs/1909.13872},
        year={2019}
    }

To embed a list of sentences:

    python embed_sentences.py --sentence-file paraphrase-at-scale/example-sentences.txt --load-file paraphrase-at-scale/model.para.lc.100.pt  --sp-model paraphrase-at-scale/paranmt.model --output-file sentence_embeds.np
    
To score a list of sentence pairs:

    python score_sentence_pairs.py --sentence-pair-file paraphrase-at-scale/example-sentences-pairs.txt --load-file paraphrase-at-scale/model.para.lc.100.pt  --sp-model paraphrase-at-scale/paranmt.model

To download and preprocess raw data for training models (both bilingual and ParaNMT), see preprocess/bilingual and preprocess/paranmt.
