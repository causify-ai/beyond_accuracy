#!/usr/bin/env python
r"""
Run the experiment based on the passed arguments:
- `batch_size`: number of time series to process in each batch 
    for parallel execution
- `num_threads`: number of parallel threads to use for processing batches
- `dst_dir`: directory to store the experiment outcomes 
- `weight_type`: type of horizon discounting weights 
    ('linear', 'exponential', 'uniform', 'hyperbolic')

# Example: run the experiment using linear weight, with 8 threads:
> python run_experiment.py \
    --batch_size 3 \
    --num_threads 8 \
    --dst_dir "./outcomes/experiment_name" \
    --weight_type "linear"
"""
import argparse
import logging
import os
from typing import List
import numpy as np
import torch
import helpers.hdbg as hdbg
import helpers.hdatetime as hdateti
import helpers.hparser as hparser
import helpers.hjoblib as hjoblib
import forecast_metric_utility as fmutil
import train_model

import fev

_LOG = logging.getLogger(__name__)


def run_experiment_and_save_outcomes(
    data, 
    selected_ids: np.ndarray,
    train_ratio: float,
    outcomes_dir,
    weight_type: str,
    *,
    nsimulations: int = 24,
    nrepetitions: int = 2,
    mult_stability: float = 0.5,
    seed: int = 42,
    **kwargs
):
    """
    Run experiments using run_on_dataset() and save the outcomes.

    :param outcomes_dir: path to save the outcomes
    Other parameters are the same as in `train_model.run_on_dataset()`.
    """
    (
        id_to_forecast_ensembles, 
        id_to_metrics, 
        id_to_model_configs,
    ) = train_model.run_on_dataset(
            data,
            selected_ids,
            train_ratio,
            weight_type = weight_type,
            nsimulations = nsimulations,
            nrepetitions = nrepetitions,
            mult_stability = mult_stability,
            seed = seed,
            **kwargs
        )
    # Save results.
    os.makedirs(outcomes_dir, exist_ok=True)
    if len(selected_ids) > 0:
        tag = f"{selected_ids[0]}_{selected_ids[-1]}"
        torch.save(
            id_to_forecast_ensembles, 
            os.path.join(outcomes_dir, f"id_to_forecast_ensembles_{tag}.pt")
        )
        torch.save(
            id_to_metrics, 
            os.path.join(outcomes_dir, f"id_to_metrics_{tag}.pt")
        )
        torch.save(
            id_to_model_configs, 
            os.path.join(outcomes_dir, f"id_to_model_configs_{tag}.pt")
        )


def _get_joblib_workload(
    data,
    batches: List[List[int]],
    train_ratio,
    outcomes_dir,
    weight_type: str,
) -> hjoblib.Workload:
    """
    Prepare the joblib workload by building all the Configs using the
    parameters from command line.

    :param batches: each batch is a list of indices (e.g., [0, 1, 2, 3]) while the param is a collection of possible batches
    """
    # Prepare one task per config to run.
    tasks = []
    for batch in batches:
        task: hjoblib.Task = (
            # args.
            (data, batch, train_ratio, outcomes_dir, weight_type),
            # kwargs.
            {},
        )
        tasks.append(task)
    #
    func_name = "_run_experiment_and_save_outcomes"
    # TODO: before we run the script, we should perhaps wrap run_on_dataset in another function that runs run_on_dataset and also saves results to disk.
    workload = (run_experiment_and_save_outcomes, func_name, tasks)
    hjoblib.validate_workload(workload)
    return workload


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Options from `run_config_list.py`.
    parser.add_argument(
        "--batch_size",
        action="store",
        # (Chutian): force `batch_size`` to be int.
        type=int,
        required=True,
        help="The number of time series to process at a time",
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        required=True,
        help="Destination dir to save the results to",
    )
    parser.add_argument(
        "--num_test",
        action="store",
        type=int,
        required=False,
        help="Number of time series to run experiment on",
    )
    parser.add_argument(
        "--weight_type",
        action="store",
        type=str,
        required=False,
        help="Type of horizon discounting weights",
    )
    # Add options related to joblib.
    parser = hparser.add_parallel_processing_arg(parser)
    parser: argparse.ArgumentParser = hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(
        verbosity=args.log_level,
        use_exec_path=True,
        # report_memory_usage=True
    )
    dst_dir = args.dst_dir
    weight_type = args.weight_type if hasattr(args, 'weight_type') else "linear"
    # I would do 3.
    batch_size = args.batch_size
    num_threads = args.num_threads
    num_test = getattr(args, 'num_test', None)
    # 1. Load the data.
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="m4_hourly",
        horizon=24,
    )
    for window in task.iter_windows():
        past_data, _ = window.get_input_data()
    id_col = "id"
    # 2. Split ts IDs in batches and generate a list of non-overlapping indices. 
    # ids_list_file_path = './outcomes/2026_01_28/ids_list_for_experiment.pkl'
    # if os.path.exists(ids_list_file_path):
    #     with open(ids_list_file_path, 'rb') as f:
    #         all_ids = pickle.load(f)
    # else:
    #     all_ids = past_data[id_col]
    if num_test:
        # Randomly selected num_test series.
        rng = np.random.default_rng(seed=42)
        all_ids = np.sort(
            rng.choice(
                past_data[id_col], 
                size=num_test, 
                replace=False,
            )
        )
    else:
        all_ids = past_data[id_col]
    # Split into batches.
    batches = [
        all_ids[i: i+batch_size] 
        for i in range(0, len(all_ids), batch_size)
    ]
    # 3. Then we call the function `run_on_dataset()` in a for-loop 
    # and iterate over batches defined above.
    # TODO(Chutian): Move this to input?
    train_ratio = 0.6
    workload = _get_joblib_workload(
        past_data,
        batches,
        train_ratio,
        dst_dir, 
        weight_type,
    )
    backend = "asyncio_threading"
    timestamp = hdateti.get_current_timestamp_as_string("naive_ET")
    log_file = os.path.join(dst_dir, f"log.{timestamp}.txt")
    dry_run = False
    incremental = False
    abort_on_error = True
    num_attempts = 1
    _LOG.info("log_file='%s'", log_file)
    hjoblib.parallel_execute(
        workload,
        dry_run,
        num_threads,
        incremental,
        abort_on_error,
        num_attempts,
        log_file,
        backend=backend,
    )


if __name__ == "__main__":
    _main(_parse())
