SUBMITIT_ERR = None

try:
    import submitit
except ImportError as err:
    _has_submitit = False
    SUBMITIT_ERR = err
else:
    _has_submitit = True


class DistributedaSyncDataCollector:
    def __init__(self, slurm_kwargs, **collector_kwargs):

        ex = submitit.AutoExecutor(folder)
        ex.update_parameters(mem_gb=1, cpus_per_task=4, timeout_min=5)
        job = ex.submit(exec_collector, **collector_kwargs)
        # get IP address of job
