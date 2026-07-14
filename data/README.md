# Dataset Instructions

## LFW (Labeled Faces in the Wild) - deep-funneled

This is the primary dataset used by the notebook to pick a source ("attacker")
face and a target ("victim") face for the impersonation attack.
 
- Download from: (https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- Extract to: `lfw-deepfunneled/lfw-deepfunneled/<Person_Name>/<Person_Name>_NNNN.jpg`
- The notebook currently picks two named individuals from this folder
  (`source_person` / `target_person` in the first code cell) and copies
  one image each into `sample_images/` as `test_image.jpg` and
  `target_image.jpg`. Edit those two names to attack a different pair.

## Sample Images

`sample_images/test_image.jpg` and `sample_images/target_image.jpg` are the
working copies actually loaded by `load_image()` and fed to `FGSMAttack`.
This directory is regenerated automatically by the notebook, so you don't
need to populate it by hand unless you want to test with your own photos.

