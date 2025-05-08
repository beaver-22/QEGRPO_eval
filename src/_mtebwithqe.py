import logging
import traceback
from datetime import datetime
from time import time
from typing import Any

import datasets
from mteb import Encoder, TaskResult, SentenceTransformerWrapper, AbsTask
from mteb.evaluation.MTEB import MTEB
from sentence_transformers import SentenceTransformer, CrossEncoder

from ._chatmodel import BaseChatModel
from .customtasks._abstaskqe import AbsTaskRetrievalWithQE

logger = logging.getLogger(__name__)


class MTEBWithQE(MTEB):

    # ToDo: Review the code: Adopted from "https://github.com/embeddings-benchmark/mteb/blob/bbbaa42618e7ceafade0e70575cb55dc4ac8211e/mteb/evaluation/MTEB.py#L296"
    @staticmethod
    def _run_eval_with_qe(
        task: AbsTask,
        retrieval_model: Encoder,
        expansion_model: BaseChatModel, # Added for custom query expansion
        split: str,
        output_folder: str | None,
        subsets_to_run: list[str] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ):
        tick = time()
        
        if not isinstance(task, AbsTaskRetrievalWithQE):
            raise TypeError(
                f"task must be of type {AbsTaskRetrievalWithQE}, got {type(task)}"
            )
        
        results = task.evaluate(
            retrieval_model,
            expansion_model, # Added for custom query expansion
            split,
            subsets_to_run=subsets_to_run,
            output_folder=output_folder,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        tock = time()
        return results, tick, tock
    
    # Adopted from "https://github.com/embeddings-benchmark/mteb/blob/bbbaa42618e7ceafade0e70575cb55dc4ac8211e/mteb/evaluation/MTEB.py#L382"
    def run_with_qe(
            self,
            retrieval_model: SentenceTransformer | Encoder,
            expansion_model: BaseChatModel,
            verbosity: int = 1,
            output_folder: str | None = "results",
            eval_splits: list[str] | None = None,
            eval_subsets: list[str] | None = None,
            overwrite_results: bool = False,
            raise_error: bool = True,
            co2_tracker: bool = False,
            encode_kwargs: dict[str, Any] = {},
            **kwargs,
    ) -> list[TaskResult]:
        print(f"Running evaluation with on {self.tasks}")
        if "batch_size" in kwargs:
            logger.warning(
                "The `batch_size` argument is deprecated and will be removed in the next release. "
                + "Please use `encode_kwargs = {'batch_size': ...}` to set the batch size instead."
            )
            encode_kwargs["batch_size"] = kwargs["batch_size"]

        # update logging to account for different levels of Verbosity (similar to the command line)

        if verbosity == 0:
            datasets.logging.set_verbosity(logging.CRITICAL)  # 40
            datasets.logging.disable_progress_bar()  # Disable progress bar
        elif verbosity == 1:
            datasets.logging.set_verbosity(logging.WARNING)
            datasets.logging.disable_progress_bar()  # Disable progress bar
        elif verbosity == 2:
            datasets.logging.set_verbosity(logging.INFO)
        elif verbosity == 3:
            datasets.logging.set_verbosity(logging.DEBUG)

        meta = self.create_model_meta(retrieval_model)
        output_path = self.create_output_folder(meta, output_folder)
        if isinstance(retrieval_model, (SentenceTransformer, CrossEncoder)):
            retrieval_model = SentenceTransformerWrapper(retrieval_model)

        ## Disable co2_tracker for API models
        if "API" in meta.framework:
            co2_tracker = False

        if output_path:
            self._save_model_metadata(meta, output_path)

        # Run selected tasks
        logger.info(f"\n\n## Evaluating {len(self.tasks)} tasks:")

        if verbosity > 0:
            self.print_selected_tasks()

        evaluation_results = []
        original_tasks = (
            self.tasks.copy()
        )  # save them in case we re-use the object (e.g. for reranking)

        # To evaluate missing splits, we keep track of the task name and the corresponding splits.
        self.last_evaluated_splits = {}

        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(
                f"\n\n********************** Evaluating {task.metadata.name} **********************"
            )

            if task.is_aggregate:
                self_ = MTEBWithQE(tasks=task.metadata.tasks)
                task_results = self_.run_with_qe(
                    retrieval_model,
                    expansion_model,
                    verbosity=verbosity - 1,
                    output_folder=output_folder,
                    eval_splits=eval_splits,
                    eval_subsets=eval_subsets,
                    overwrite_results=overwrite_results,
                    raise_error=raise_error,
                    co2_tracker=co2_tracker,
                    encode_kwargs=encode_kwargs,
                    **kwargs,
                )
                new_results = task.combine_task_results(task_results)
                evaluation_results.append(new_results)

                if output_path:
                    save_path = output_path / f"{task.metadata.name}.json"
                    new_results.to_disk(save_path)
                del self.tasks[0]
                continue

            if "bm25s" in meta.name and task.metadata.type != "Retrieval":
                logger.warning(
                    f"bm25s only supports Retrieval tasks, but the task type is {task.metadata.type}. Skipping task."
                )
                del self.tasks[0]  # empty memory
                continue

            # NOTE: skip evaluation if the model does not support all of the task's modalities.
            # If the model covers more than the task's modalities, evaluation will still be run.
            sorted_task_modalities = sorted(task.metadata.modalities)
            if meta.modalities is not None and any(
                    m not in meta.modalities for m in sorted_task_modalities
            ):
                logger.info(
                    f"{meta.name} only supports {meta.modalities}, but the task modalities are {sorted_task_modalities}."
                )
                del self.tasks[0]  # empty memory
                continue

            task_eval_splits = (
                eval_splits if eval_splits is not None else task.eval_splits
            )
            task_subsets = (
                task.hf_subsets
                if task.hf_subsets
                else list(task.metadata.hf_subsets_to_langscripts.keys())
            )

            existing_results = None
            save_path = None
            final_splits_to_run = task_eval_splits
            missing_evaluations = self._get_missing_evaluations(
                existing_results,
                task_eval_splits,
                task_subsets,
                eval_subsets,
            )

            if output_path:
                save_path = output_path / f"{task.metadata.name}{task.save_suffix}.json"
                if save_path.exists():
                    existing_results = TaskResult.from_disk(save_path)

                    # Unified call to get missing splits and subsets
                    missing_evaluations = self._get_missing_evaluations(
                        existing_results,
                        task_eval_splits,
                        task_subsets,
                        eval_subsets,
                    )

                    if overwrite_results:
                        final_splits_to_run = task_eval_splits
                    else:
                        # Determine final splits to run
                        final_splits_to_run = []
                        # We need to run any split that is fully missing or has missing subsets
                        for sp, info in missing_evaluations.items():
                            if info["whole_split_missing"] or info["missing_subsets"]:
                                final_splits_to_run.append(sp)

                    if not overwrite_results and len(final_splits_to_run) == 0:
                        logger.info(
                            f"{task.metadata.name} results already exists. Loading results from disk."
                            f" Set overwrite_results=True to overwrite or `--overwrite`."
                        )
                        evaluation_results.append(existing_results)
                        del self.tasks[0]  # empty memory
                        continue

            # If no splits need to be run and results exist, skip
            if not final_splits_to_run:
                if existing_results is not None:
                    evaluation_results.append(existing_results)
                else:
                    logger.info(
                        f"No splits to evaluate for {task.metadata.name}. Skipping evaluation."
                    )
                self.last_evaluated_splits[task.metadata.name] = []
                del self.tasks[0]
                continue

            try:
                task.check_if_dataset_is_superseded()
                task.load_data(**kwargs)

                task_results = {}
                evaluation_time = 0
                kg_co2_emissions: int | None = 0 if co2_tracker else None

                self.last_evaluated_splits[task.metadata.name] = []

                for split in final_splits_to_run:
                    info = missing_evaluations[split]

                    # Determine subsets to run for this split
                    # If the whole split is missing, run all required subsets
                    # If only some subsets are missing, run only those
                    subsets_to_run = (
                        info["missing_subsets"]
                        if not overwrite_results
                        else (eval_subsets or task_subsets)
                    )

                    if (
                            info["whole_split_missing"] or overwrite_results
                    ) and task_subsets is None:
                        subsets_to_run = ["default"]

                    if co2_tracker:
                        try:
                            from codecarbon import EmissionsTracker
                        except ImportError:
                            raise ImportError(
                                "To use the CO2 emissions tracker, please install codecarbon using 'pip install codecarbon'"
                            )
                        with EmissionsTracker(
                                save_to_file=False, save_to_api=False, logging_logger=logger
                        ) as tracker:
                            results, tick, tock = self._run_eval(
                                task,
                                retrieval_model,
                                split,
                                output_folder,
                                encode_kwargs=encode_kwargs,
                                subsets_to_run=subsets_to_run,
                                **kwargs,
                            )

                        kg_co2_emissions += (
                            tracker.final_emissions
                        )  # expressed as kilograms of CO₂-equivalents
                    else:
                        # Todo: Review Custom Code: Adopted from "https://github.com/embeddings-benchmark/mteb/blob/bbbaa42618e7ceafade0e70575cb55dc4ac8211e/mteb/evaluation/MTEB.py#L611""
                        results, tick, tock = self._run_eval_with_qe(
                            task,
                            retrieval_model,
                            expansion_model,
                            split,
                            output_folder,
                            subsets_to_run=subsets_to_run,
                            encode_kwargs=encode_kwargs,
                            **kwargs,
                        )

                    logger.info(
                        f"Evaluation for {task.metadata_dict['name']} on {split} took {tock - tick:.2f} seconds"
                    )
                    evaluation_time += tock - tick

                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {task_results[split]}")

                    self.last_evaluated_splits[task.metadata.name].append(split)

                # Create new TaskResult
                new_results = TaskResult.from_task_results(
                    task,
                    task_results,
                    evaluation_time=evaluation_time,
                    kg_co2_emissions=kg_co2_emissions,
                )

                # Merge with existing if needed
                if output_path and save_path.exists():
                    existing_results = TaskResult.from_disk(save_path)
                if existing_results:
                    merged_results = self._merge_results(existing_results, new_results)
                else:
                    merged_results = new_results

                if output_path:
                    merged_results.to_disk(save_path)

                evaluation_results.append(merged_results)

            except Exception as e:
                logger.error(
                    f"Error while evaluating {task.metadata_dict['name']}: {e}"
                )
                if raise_error:
                    raise e
                logger.error(
                    f"Please check all the error logs at: {self.err_logs_path}"
                )
                with open(self.err_logs_path, "a") as f_out:
                    f_out.write(f"{datetime.now()} >>> {task.metadata_dict['name']}\n")
                    f_out.write(traceback.format_exc())
                    f_out.write("\n\n")

            # empty memory
            del self.tasks[0]

        self.tasks = original_tasks
        return evaluation_results
