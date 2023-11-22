from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, LinearLR, StepLR

SCHED_STEP_PARAM = ['ReduceLROnPlateau']

dict_optimizer = {
    "Adam": Adam
}

dict_optimizer_scheduler = {
    "ReduceLROnPlateau": {'class': ReduceLROnPlateau,
                          'params': {'mode': 'min', 'factor': 0.75, 'patience': 30}
                          },
    "MultiStepLR": {'class': MultiStepLR,
                    'params': {'milestones': [30, 80], 'gamma': 0.1}
                    },
    "LinearLR": {'class': LinearLR,
                 'params': {'start_factor': 0.5, 'total_iters': 4}
                 },
    "StepLR": {'class': StepLR,
               'params': {'step_size': 30, 'gamma': 0.1}
               },
}
