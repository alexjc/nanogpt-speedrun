"""
Code originally by Andrej Karpathy:
https://github.com/karpathy/llm.c/blob/master/dev/data/hellaswag.py

Downloads and evaluates HellaSwag in Python.
This then acts as the reference file for llm.c
Also writes the data (tokens, labels) to .bin files for parallel evaluation in C.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import math
import requests
import tiktoken
import tokenmonster
from tqdm import tqdm
import torch
from torch.nn import functional as F

# -----------------------------------------------------------------------------

"""
Common utilities for the datasets
"""

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


# This variable holds the tokenization function that will be setup in main().
enc_encode = None


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc_encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc_encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens + [50256])
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens) + [0])
        data["ending_tokens"].append(end_tokens)

    # concatenate all tokens and masks into 1D arrays
    tokens = torch.cat([torch.tensor(row) for row in tok_rows], dim=0)
    mask = torch.cat([torch.tensor(row) for row in mask_rows], dim=0)
    # pad to next multiple of 32
    pad_to = math.ceil(len(tokens) / 32) * 32
    tokens = F.pad(tokens, (0, pad_to - len(tokens)))
    mask = F.pad(mask, (0, pad_to - len(mask)))

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        examples = [json.loads(line) for line in f]
        import random
        random.seed(12345)
        random.shuffle(examples)
        for e in examples:
            yield e

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32

    # NOTE: This assumes that train_gpt2.py has a main() function and doesn't do anything when imported as a module.
    from train_gpt2 import GPT, GPTConfig
    model = GPT(GPTConfig())
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load(model_type, map_location="cpu")["model"].items()})
    model.to(device)

    datas = []
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        datas.append(data)

        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).squeeze(0)

        # evaluate the autoregressive loss at all positions
        shift_logits = logits[:-1, :].contiguous()
        shift_tokens = tokens[1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')

        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = mask[1:]
        masked_shift_losses = shift_losses * shift_mask

        # Find indices of EOT tokens (50256)
        eot_indices = (tokens == 50256).nonzero().squeeze()

        # Calculate ranges between EOT tokens for the 4 samples
        ranges = []
        start = 0
        for i in range(4):
            end = eot_indices[i].item() if i < len(eot_indices) else len(tokens[0])
            assert tokens[end] == 50256
            ranges.append((start, end-1)) # exclude EOT token
            start = end + 1

        # Initialize arrays to store losses for each completion
        sum_loss = torch.empty(4, device=device)
        avg_loss = torch.empty(4, device=device)

        # Calculate sum and average loss for each completion range
        for i, (start, end) in enumerate(ranges):
            assert start < end

            # Adjust start and end to account for the shift
            shift_start = max(0, start - 1)
            shift_end = end - 1

            # Get losses and mask for this range
            range_losses = masked_shift_losses[shift_start:shift_end]
            range_mask = shift_mask[shift_start:shift_end]
            
            assert torch.all((range_losses != 0) == range_mask), "Mismatch between non-zero losses and mask positions"

            # Sum losses and divide by number of tokens for average
            sum_loss[i] = range_losses.sum()
            avg_loss[i] = sum_loss[i] / range_mask.sum()

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        if num_total % 100 == 0:
            print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}", end="\r")

        # debug: pretty print a few examples, and the losses in each case
        # if num_total % 100 == 1:
        #     print("---")
        #     print(f"Context:\n {example['ctx']}")
        #     print(f"Endings:")
        #     for i, end in enumerate(example["endings"]):
        #         print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
        #     print(f"predicted: {pred_norm}, actual: {label}")

    print('MODEL', model_type)
    print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", type=str, nargs="+", help="the model .pt files to evaluate")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="the device to use")
    parser.add_argument("-t", "--tokenizer", type=str, default="tokenmonster", choices=["tiktoken", "tokenmonster"], help="tokenizer to use")
    parser.add_argument("-v", "--vocabulary", type=str, default="./english-50256-balanced-v2", help="vocabulary file to use") 
    args = parser.parse_args()

    if args.tokenizer == "tiktoken":
        enc = tiktoken.get_encoding("gpt2")
        enc_encode = lambda text: enc.encode(text)
    if args.tokenizer == "tokenmonster":
        tokenmonster.set_local_directory("data/")
        enc = tokenmonster.load(args.vocabulary)
        enc_encode = lambda text: [int(t) for t in enc.tokenize(text)]

    for model in sorted(args.models):
        evaluate(model, args.device)
