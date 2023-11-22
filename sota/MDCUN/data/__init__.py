from .alsace import FusionDatasetALSACE as Alsace
from .wv_qb_geofen import WV3_QB_Geofen
from .QuickBirdTest import QuickBirdTest
dict_dataset = {
    # "CAVE": Cave,
    "ALSACE": Alsace,
    "WorldView3": WV3_QB_Geofen,
    "QuickBird": WV3_QB_Geofen,
    "Geofen": WV3_QB_Geofen,
    "QuickBirdTest": QuickBirdTest
    # "CHIKUSEI": Chikusei,
    # "HARVARD": Harvard,
    # "PRISMA": Prisma
}