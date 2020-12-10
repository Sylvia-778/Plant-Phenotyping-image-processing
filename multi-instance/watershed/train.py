# created by Liqi Jiang

from task32 import sdb_score
import argparse
from os import listdir
from os.path import join


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training DeepColoring on CVPPP dataset.')
    # Path to the folder

    parser.add_argument('-s',
                        '--basepath',
                        type=str,
                        action='store',
                        help="Path to CVPPP A1 dataset folder",
                        default="Plant/Ara2013-Canon/",
                        required=False)

    args = parser.parse_args()

    basepath = args.basepath

    rgbs = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])
    score_sum = 0
    for (i, rgb) in enumerate(rgbs):
        if rgbs[i][:-8] == labels[i][:-10]:
            score_sum += sdb_score(rgb, labels[i])
        else:
            print("rgb image name", rgbs[i][:-8],"is not equal to label image name", labels[i][:-10])
    sdb = score_sum / len(rgbs)
    print("The Symmetric Best Dice for", basepath[6:-1], "is", sdb)
