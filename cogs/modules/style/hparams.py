import math
import multiprocessing as mp

import torch


class HParams(object):
    def __init__(self, customHParams, **kwargs):
        super(HParams, self).__init__()
        self.init_params(customHParams, **kwargs)

    def init_params(self, customHParams, **kwargs):
        self.customHParams = list(customHParams.keys())

        self.load_state(kwargs)
        self.load_state(customHParams)

    def set_params(self, args=None, **custom):
        custom = parse_command_line_args(args, **custom)
        self.init_params(custom)

    def save_state(self):
        state = {}
        for key in self.__dict__:
            state[key] = self.__dict__[key]
        return state

    def load_state(self, state):
        for key in state:
            setattr(self, key, state[key])
            self.__dict__[key] = state[key]


def parse_command_line_args(args=None, **custom):
    # Parse command-line hparam arguments
    if args is not None:
        key_vals = args.split(",")
        for kv in key_vals:
            k, v = kv.split("=")

            # Handle numerical values
            try:
                v = float(v)
                if int(v)==v:
                    v = int(v)
            except:
                # Handle boolean values
                if v=="False":
                    v = False
                elif v=="True":
                    v = True
                pass
            custom[k] = v
    return custom


def initParams(args=None, **custom):
    custom = parse_command_line_args(args, **custom)

    return HParams(custom,
        ## VLAE
        latent_size = 32,
        resize_shape = 28,

        ## Meta
        run_name = "",
        pretrained = True, # For use with torch models
        pretrained_out = 10, # For use with torch models
        use_gpu = torch.cuda.is_available(),
        clear_gpu_cache = False,
        gpu_index = 0,
        workers = mp.cpu_count(),
        multi_gpu = False,
        multi_gpu_indeces = list(range(torch.cuda.device_count())),
        fp16 = False,
        pin_memory = False,
        fixed_seed = True,
        max_iter = math.inf,
        server = "",
        logger = None,
        dataloader_shuffle = [True, True, False],
        debug=False,
        note=None,

        ## Training
        grad_acc_multiplier = 1,
        clip = 0,
        lr = 1e-3,
        epochs = 50,
        batch_size = 1,
        grad_clip= 1,
        optim = "SGD",
        amsgrad = False,
        weight_decay = 0,
        momentum = 0.5,
        lr_decay = False,
        lr_decay_gamma = 0.5,
        lr_decay_step_size = 100000,
        noam_decay = False,
        resume_ckpt_dir = False,
        gc_collect = True, # Wether or not to run gc.collect() after every checkpoint

        ## Validating / Checkpointing
        valPatience = 10,
        ckpt_interval = 100,
        val_interval = 100,
        ckpt_at_zero = False, # If a validation/checkpoint should be done at training iteration 0
        ckpt_dir = "checkpoints",
        max_ckpts = 5,
        max_viz_count = 5, # How many viz images to keep at any time (at either side: start/latest)
        backup_every_n_ckpt = None # Large number of iterations to keep a copy of the ckpts between
    )
