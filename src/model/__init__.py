from classic_methods import dict_classic_methods
from .NLBPUN import NLBPUN
from sota import sota_dict

dict_model = dict(
    NLBPUN=NLBPUN,
    **sota_dict,
    **dict_classic_methods
)
