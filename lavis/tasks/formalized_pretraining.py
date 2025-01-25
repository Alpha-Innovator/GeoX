from lavis.common.registry import registry
from lavis.tasks import BaseTask

@registry.register_task("formalized_pretraining")
class FormalizedPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
