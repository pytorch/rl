from torch import Tensor
class Logger:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.experiment = self._create_experiment()
    def _create_experiment(self):
        pass
    def log_scalar(self, name: str, value: float, step: int=None):
        pass
    def log_video(self, name: str, video: Tensor, step: int=None, **kwargs):
        pass
    def log_hparams(self, cfg):
        pass
    def __repr__(self) -> str:
        return self.experiment.__repr__()

class TensorboardLogger(Logger):
    def __init__(self, exp_name: str):
        super().__init__(exp_name=exp_name)
        self.log_dir = self.experiment.log_dir

        self._has_imported_moviepy = False
        
    def _create_experiment(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except:
            raise ImportError("torch.utils.tensorboard could not be imported")
        return SummaryWriter(log_dir = self.exp_name)

    def log_scalar(self, name: str, value: float, step: int=None):
        self.experiment.add_scalar(name, value, global_step=step)
    
    def log_video(self, name: str, video: Tensor, step: int=None, **kwargs):
        if not self._has_imported_moviepy:
            try:
                import moviepy  # noqa
                self._has_imported_moviepy = True
            except ImportError:
                raise Exception("moviepy not found, videos cannot be logged with TensorboardLogger")
        self.experiment.add_video(
            tag=name,
            vid_tensor=video,
            global_step=step,
            **kwargs,
        )
    def log_hparams(self, cfg: "DictConfig"):
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(cfg).items())])
        self.experiment.add_text("hparams", txt)