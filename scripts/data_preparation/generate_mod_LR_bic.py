import os
import sys
import cv2
import numpy as np
import argparse

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from utils.utils_deg import imresize
except ImportError:
    pass


def generate_mod_LR_bic(up_scale, mod_scale, sourcedir, savedir):

    saveHRpath = os.path.join(savedir, "HR", "x" + str(mod_scale))
    saveLRpath = os.path.join(savedir, "LR", "x" + str(up_scale))

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, "HR")):
        os.mkdir(os.path.join(savedir, "HR"))
    if not os.path.isdir(os.path.join(savedir, "LR")):
        os.mkdir(os.path.join(savedir, "LR"))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print("It will cover " + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print("It will cover " + str(saveLRpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith(".png") or f.endswith(".jpg")]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width, :]
        else:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width]
        
        # LR
        image_LR = imresize(image_HR, 1 / up_scale, True)
        
        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate modified LR and bicubic HR images")
    parser.add_argument("--mod_scale", type=int, default=2, help="Modification scale factor")
    parser.add_argument("--up_scale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--sourcedir", type=str, default="../../testsets/Set5/original", help="Source directory")
    parser.add_argument("--savedir", type=str, default="../../testsets/Set5", help="Save directory")
    args = parser.parse_args()
    generate_mod_LR_bic(args.up_scale, args.mod_scale, args.sourcedir, args.savedir)


"""usage:

in terminal:
pytho3 generate_mod_LR_bic.py \
  --sourcedir /path/to/input_folder \
  --savedir /path/to/save_folder \
  --mod_scale 2 \
  --up_scale 2
in cmd:
python generate_mod_LR_bic.py ^
--sourcedir /path/to/input_folder ^
--savedir /path/to/save_folder ^
--mod_scale 2 ^
--up_scale 2 
in powershell:
python generate_mod_LR_bic.py `
--sourcedir /path/to/input_folder `
--savedir /path/to/save_folder `
--mod_scale 2 `
--up_scale 2 
"""
