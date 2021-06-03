import argparse
import numpy as np
from sacremoses import MosesTokenizer
from models import load_model
from utils import Example
import time

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def embed(params, batcher, sentences):
    results = []
    for ii in range(0, len(sentences), params.batch_size):
        batch1 = sentences[ii:ii + params.batch_size]
        results.extend(batcher(params, batch1))
    return np.vstack(results)

def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

def embed_all(args, model):

    entok = MosesTokenizer(lang='en')

    from argparse import Namespace

    new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                     params=args, model=model, lower_case=model.args.lower_case,
                     tokenize=model.args.tokenize)

    fin = open(args.sentence_file, 'r', errors='surrogateescape')
    fout = open(args.output_file, mode='wb')
    n = 0
    t = time.time()
    for sentences in buffered_read(fin, 10000):
        embed(new_args, batcher, sentences).tofile(fout)
        n += len(sentences)
        if n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    print('\r - Encoder: {:d} sentences'.format(n), end='')
    t = int(time.time() - t)
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))
    fin.close()
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load-file", help="path to saved model")
    parser.add_argument("--sp-model", help="sentencepiece model to use")
    parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
    parser.add_argument("--sentence-file", help="sentence file")
    parser.add_argument("--output-file", help="prefix for output numpy file")
    args = parser.parse_args()

    model, _ = load_model(None, args)
    print(model.args)
    model.eval()
    embed_all(args, model)
