class Logger:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.experiment = self._create_experiment()
    def _create_experiment(self):
        pass
    def log_scalar(self, name, value, step=None):
        pass
    def log_video(self):
        pass
    def log_hparams(self):
        pass
    def __repr__(self) -> str:
        return self.experiment.__repr__()

class TensorboardLogger(Logger):
    def __init__(self, exp_name: str):
        super().__init__(self, exp_name)

        self._has_imported_moviepy = False
        
    def _create_experiment(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except:
            raise ImportError("torch.utils.tensorboard could not be imported")
        return SummaryWriter(log_dir = self.exp_name)

    def log_scalar(self, name: str, value: float, step=None):
        self.experiment.add_scalar(name, value, global_step=step)
    
    def log_videos(self, name, video, step=None, **kwargs):
        if not self._has_moviepy:
            try:
                import moviepy  # noqa
                self._has_moviepy = True
            except ImportError:
                raise Exception("moviepy not found, videos cannot be logged with TensorboardLogger")
        self.experiment.add_video(
            tag=name,
            vid_tensor=video,
            global_step=self,
            **self.kwargs,
        )
    def log_hparams(self, cfg):
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(cfg).items())])
        self.experiment.add_text("hparams", txt)