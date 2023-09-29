import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model_search, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
from nets.attention_model import set_decode_type
import csv
import warnings 



def eval_dataset(dataset_path, width, softmax_temp, opts):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='torch.serialization')
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model_search(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
    results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
    return

def _eval_dataset(model, dataset, width, softmax_temp, opts, device):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='torch.serialization')

    model.to(device)

    model.set_decode_type("greedy")

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    results1 = []
    import datetime
    a = datetime.datetime.now()
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            set_decode_type(model, "greedy")
            model.train()
            pis, costs = model(batch, beam_size=opts.beam_size, fst=1)
            # print(costs)
            # sequences, costs = model.sample_many(batch)
            results.append(costs)
            # print(costs)
            # print(pi)
            # costs, pi  = model.get_costs(batch)

    # results = torch.cat(results, 0)


    b = datetime.datetime.now()
    delta = b - a

    # print (results.mean().item(), delta)

    results1.append((costs, pis, delta))

    output_file = opts.results_save

    # Check if the output file already exists
    file_exists = os.path.isfile(output_file)

    # Define the initial cluster number
    initial_cluster_number = 0

    # Determine the new cluster number based on the previous value in the file (if it exists)
    if file_exists:
        with open(output_file, 'r') as file:
            reader = csv.reader(file)
            last_row = list(reader)[-1] if any(reader) else None  # Get the last row in the file if it exists

        if last_row and len(last_row) >= 4:  # Check if last_row exists and has enough elements
            previous_cluster_number = int(last_row[3])  # Assuming the cluster number is in the 4th column
            new_cluster_number = previous_cluster_number + 1
        else:
            new_cluster_number = initial_cluster_number
    else:
        new_cluster_number = initial_cluster_number

    # Prepare the data to write to the CSV file, including the cluster number
    results_with_cluster = [(costs, pis.tolist(), duration, new_cluster_number) for cost, pi, duration in results1]

    # Save the results to a CSV file
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist or is empty, write the header
        if not file_exists or file_exists and not last_row:
            writer.writerow(['Cost', 'Sequence', 'Duration', 'Cluster Number'])

        writer.writerows(results_with_cluster)  # Append data rows with the cluster number

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")


    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")


    parser.add_argument('--beam_size', type=int, help="beam size")
    parser.add_argument('--results_save')

    opts = parser.parse_args()

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
