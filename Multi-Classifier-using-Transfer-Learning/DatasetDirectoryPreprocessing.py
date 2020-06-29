import os
import shutil

base_dir = "data/Monkey-Species/"

train_dir = os.path.join(base_dir, "training/")
validation_dir = os.path.join(base_dir, "validation/")
test_dir = "data/testing/"

class_dict = {"n0" : "mantled howler",
              "n1" : "patas monkey",
              "n2" : "bald uakari",
              "n3" : "japanese macaque",
              "n4" : "pygmy marmoset",
              "n5" : "white headed capuchin",
              "n6" : "silvery marmoset",
              "n7" : "common squirrel monkey",
              "n8" : "black headed night monkey",
              "n9" : "nilgiri langur"}

num_classes = len(class_dict)
print("Number of classes: ", num_classes)