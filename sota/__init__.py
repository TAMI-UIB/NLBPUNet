from sota.GPPNN.models.GPPNN import GPPNN
from sota.MDCUN.model.pan_unfolding_v4 import pan_unfolding
from sota.NLRNET.net.nlrnet.NLRNet import NLRNet_adapted as NLRNet

sota_dict = {
    'GPPNN': GPPNN,
    'MDCUN': pan_unfolding,
    'NLRNet': NLRNet,
}
