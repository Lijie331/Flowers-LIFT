import os
from .lt_data import LT_Dataset


class Oxford_Flowers(LT_Dataset):
    """Oxford Flowers 102 Dataset"""
    train_txt = "./oxford_flowers_train.txt"
    test_txt = "./oxford_flowers_test.txt"
    
    # 102个花卉类别的名称
    classnames = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
        "english marigold", "tiger lily", "moon orchid", "bird of paradise",
        "monkshood", "globe thistle", "snapdragon", "coltsfoot",
        "king protea", "spear thistle", "yellow iris", "globe flower",
        "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
        "fire lily", "pincushion mushroom", "fritillary", "red ginger",
        "sunflower", "lily", "calendula", "marsh orchid",
        "artichoke", "hibiscus", "lotus lotus", "foxtail lily",
        "clematis", "hibiscus", "larkspur", "carnation",
        "garden phlox", "love-in-the-mist", "cosmos", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "siam tulip", "lenten rose",
        "barbeton daisy", "daffodil", "magnolia", "cyclamen",
        "watercress", "monkshood", "arts shawl", "kingfisher",
        "corn poppy", "prince of wales feathers", "gypsophila", "ardtemis",
        "busy lizzie", "bromelia", "magnolia", "mexican petunia",
        "bougainvillea", "camellia", "mallow", "mexican hat",
        "geranium", "pentas", "bee balm", "balloon flower",
        "oxeye daisy", "black-eyed susan", "cobaea", "blanket flower",
        "trumpet creeper", "blackberry lily", "common tulip", "wild rose",
        "thorn apple", "morning glory", "passion flower", "lotus",
        "toad lily", "anemone", "frangipani", "plumeria",
        "hippeastrum", "blue poppy", "celandine", "tree poppy",
        "azalea", "flowering cherry", "indian strawberry", "frangipani",
        "magenta spider lily", "gaillardia", "yarrow", "colchicum",
        "mexican sunflower", "oxeye daisy", "gardenias", "marigold",
        "petunia", "california poppy", "canna lily", "osteospermum",
        "california poppy", "snapdragon", "camellia", "impatiens",
        "begonia", "lantana", "verbena", "wedelia",
        "yellow sunflower", "hirsute viola", "canna", "petunia"
    ]

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)
