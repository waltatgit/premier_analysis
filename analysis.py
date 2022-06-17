'''This script generates bootstrap CIs for the models and metrics'''
import argparse
import os
from functools import partial
import pandas as pd
import pickle
import argparse
import os

import tools.analysis as ta
import tools.multi as tm

import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--parallel',
        help=
        "Compute BCa CIs in parallel (can speed up execution time but requires more memory and CPU usage)",
        type="bool")
    parser.set_defaults(parallel=True)
    parser.add_argument('--n_boot',
                        type=int,
                        default=100,
                        help='How many bootstrap samples to use')
    parser.add_argument(
        '--processes',
        type=int,
        default=8,
        help=
        'How many processes to use in the Pool (ignored if parallel is false)')
    parser.add_argument("--out_dir",
                        type=str,
                        help="output directory (optional)")
    parser.add_argument("--outcome",
                        type=str,
                        default=["misa_pt", "multi_class", "death", "icu"],
                        nargs="+",
                        choices=["misa_pt", "multi_class", "death", "icu"],
                        help="which outcome to compute CIs for (default: all)")
    parser.add_argument('--model',
                        type=str,
                        default='',
                        help='which models to evaluate; must either be "all" \
                    or a single column name from the preds files, like \
                    "lgr_d1" or "lstm"')
    args = parser.parse_args()

    # Globals
    N_BOOT = args.n_boot
    PROCESSES = args.processes
    OUTCOME = args.outcome
    PARALLEL = args.parallel
    MODEL = args.model

    # Choose which CI function to use
    if PARALLEL:
        boot_cis = partial(tm.boot_cis, processes=PROCESSES)
    else:
        boot_cis = ta.boot_cis

    # Setting the directories
    pwd = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.abspath(os.path.join(pwd, "..", "output", ""))

    # Over-ride default path if one is provided
    if args.out_dir is not None:
        output_dir = os.path.abspath(args.out_dir)

    stats_dir = os.path.abspath(os.path.join(output_dir, "analysis", ""))
    probs_dir = os.path.abspath(os.path.join(stats_dir, "probs", ""))

    # Path where the metrics will be written
    ci_file = os.path.join(stats_dir, "cis.xlsx")

    # Checking prob files
    prob_files = os.listdir(probs_dir)

    cis = []

    for i, outcome in enumerate(OUTCOME):
        outcome_cis = []

        # Importing the predictions
        preds = pd.read_csv(os.path.join(stats_dir, outcome + '_preds.csv'))

        # Take only single model if specified, else
        # pull a list of all models from the preds file dynamically
        if MODEL != "":
            mods = MODEL
        else:
            mods = [col for col in preds.columns if "_pred" in col]
            mods = [re.sub("_pred", "", mod) for mod in mods]

        for mod in mods:
            mod_prob_file = mod + '_' + outcome + '.pkl'

            if mod_prob_file in prob_files:
                # In the multi_class case, we've been writing pkls
                with open(os.path.join(probs_dir, mod_prob_file), 'rb') as f:
                    probs_dict = pickle.load(f)

                    cutpoint = probs_dict['cutpoint']
                    guesses = probs_dict['probs']
            else:
                # Otherwise the probs will be in the excel file
                cutpoint = 0.5
                guesses = preds[mod + '_pred']

            # Compute CIs model-by-model
            ci = boot_cis(targets=preds[outcome],
                          guesses=guesses,
                          cutpoint=cutpoint,
                          n=N_BOOT)
            # Append to outcome CI list
            outcome_cis.append(ci)

        # Append all outcome CIs to master list
        cis.append(ta.merge_ci_list(outcome_cis, mod_names=mods, round=2))

    # Writing all the confidence intervals to disk
    append_flag = "a" if os.path.exists(ci_file) else "w"

    with pd.ExcelWriter(ci_file, mode=append_flag) as writer:
        for i, outcome in enumerate(OUTCOME):
            cis[i].to_excel(writer, sheet_name=outcome)
        writer.save()