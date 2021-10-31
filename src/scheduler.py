# WWIII AutoTVM V.S. Auto-Scheduler (a.k.a Ansor)
import os
import argparse

import tvm
from tvm import relay, auto_scheduler
# from utils import get_network, make_network_key


# 1st candidate: AutoTVM
def auto_tvm_tune(tasks, log_filename, n_trial):
    tmp_log_file = log_filename + ".tmp"
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        tuner = tvm.autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type='curve')
        if os.path.isfile(tmp_log_file):
            tuner.load_history(tvm.autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner.tune(
            n_trial=n_trial,
            early_stopping=600,
            measure_option=tvm.autotvm.measure_option(
                builder=tvm.autotvm.LocalBuilder(timeout=10),
                runner=tvm.autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)),
            callbacks=[
                tvm.autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                tvm.autotvm.callback.log_to_file(tmp_log_file)
            ])

    # pick best records to a cache file
    tvm.autotvm.record.pick_best(tmp_log_file, log_filename)

# 2nd candidate: Auto-Scheduler
def auto_scheduler_tune(tasks, log_filename, n_trial):

    tuning_opt = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_filename)],
    )

    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)