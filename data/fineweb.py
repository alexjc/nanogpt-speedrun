"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import hashlib
import argparse
import multiprocessing as mp
import numpy as np
import tokenmonster
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
if args.version == "10B":
    local_dir = "fineweb-tokmon-10B"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "fineweb-tokmon-100B"
    remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir, "english-28416-balanced-v1")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# check the vocabulary file that's expected
VOCAB_HASH_SHA256 = "7cf495836f9112c1d379d298d7a77f88fc43ec6717760cc8384f56905291e473"
with open(os.path.join(os.path.dirname(__file__), "english-28416-balanced-v1.vocab"), "rb") as f:
    assert hashlib.sha256(f.read()).hexdigest() == VOCAB_HASH_SHA256

# init the tokenizer
tokenmonster.set_local_directory(os.path.dirname(__file__))
enc = tokenmonster.load(f"english-28416-balanced-v1.vocab")
eot = 28415
eof = 65535

assert len(enc.get_dictionary()) == 28415


text_overflow = []

def tokenize(docs):
    # NOTE: To ensure that the validation set matches exactly the original GPT-2 tokens, we hard-code a
    # boundary here and return a special token (65535) to suggest that a new file should be started.
    texts = text_overflow
    text_overflow.clear()
    is_boundary = False
    for i, t in enumerate(docs["text"]):
        if is_boundary:
            text_overflow.append(t)
            continue

        # This is the boundary of the entire validation file.
        if docs["id"][i] == "<urn:uuid:86393c58-6372-41bc-8beb-27be939c8eec>":
            assert docs["url"][i] == "http://www.navhindtimes.in/iwatch/lighting-ramp-starry-eyed-dreams"
            assert t[:897].endswith('there are times when we are so into the ramp walk that even when it has been called a day, we go back to our rooms')
            texts.append(t[:897])
            text_overflow.append(t[897:])
            is_boundary = True
        ## This is the boundary of the validation tokens used.
        # if docs["id"][i] == "<urn:uuid:c4b8bb84-3c4c-4dc6-8d67-fa445a0df774>":
        #     assert docs["url"][i] == "http://www.bellaonline.com/articles/art55186.asp"
        #     assert t[:1351].endswith("‘Please help us or we will die. January 26, 8:00 A.M.’\n")
        #     texts.append(t[:1351])
        #     is_boundary = True
        else:
            texts.append(t)

    tokens = []
    for t in enc.tokenize(texts):
        tokens.append(eot)
        tokens.extend(t)

    if is_boundary:
        tokens.append(eof)

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def main():
    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    # NOTE: Multi-processing is disabled because it causes TokenMonster inter-process communication to hang.
    with mp.Pool(processes=1) as _:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in map(tokenize, fw.iter(batch_size=1024)):
            # if the boundary of this binary file is forced with an EOF token, make a new one.
            is_end_of_file = False
            if tokens[-1] == eof:
                is_end_of_file = True
                tokens = tokens[:-1]

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < args.shard_size and not is_end_of_file:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb-tokmon_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                if is_end_of_file:
                    remainder = 0
                    progress_bar.update(len(tokens))
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    write_datafile(filename, all_tokens_np[:token_count+len(tokens)])
                    assert split == "val"
                else:
                    remainder = args.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb-tokmon_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
