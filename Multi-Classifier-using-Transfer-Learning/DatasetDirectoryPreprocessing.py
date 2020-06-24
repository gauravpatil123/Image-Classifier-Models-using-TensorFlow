import os
import shutil

base_dir = "data/PokemonData/"

training_portion = 0.6
validation_portion = 0.3
testing_portion = 0.1

train_dir = "data/training/" 
validation_dir = "data/validation/" 
test_dir = "data/testing/"

os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

class_list = os.listdir(base_dir)
class_list = sorted(class_list)
try:
    class_list.remove(".DS_Store")
except:
    print("No .DS_Store file")
num_classes = len(class_list)
print("Number of classes: ", num_classes)
# print(class_list)

total_training_images = 0
total_validation_images = 0
total_testing_images = 0

# preprocessing and splitting datatsets
for i in range(num_classes):
    folder_name = class_list[i]
    path = os.path.join(base_dir, folder_name)
    
    image_list = os.listdir(path)
    image_list = sorted(image_list)
    num_class_images = len(image_list)
    num_train_images = int(training_portion * num_class_images)
    num_validation_images = int(validation_portion * num_class_images)
    num_test_images = num_class_images - (num_train_images + num_validation_images)

    # print(len(image_list))
    train_class = os.path.join(train_dir, folder_name)
    os.mkdir(train_class)
    validation_class = os.path.join(validation_dir, folder_name)
    os.mkdir(validation_class)
    test_class = os.path.join(test_dir, folder_name)
    os.mkdir(test_class)

    # splitting datatsets
    train_images = image_list[0:num_train_images]
    validation_images = image_list[num_train_images:(num_train_images + num_validation_images)]
    test_images = image_list[(num_train_images + num_validation_images):]
    total_training_images += len(train_images)
    total_validation_images += len(validation_images)
    total_testing_images += len(test_images)

    for image in train_images:
        source_path = os.path.join(path, image)
        destination_path = os.path.join(train_class, image)
        shutil.copyfile(source_path, destination_path)

    for image in validation_images:
        source_path = os.path.join(path, image)
        destination_path = os.path.join(validation_class, image)
        shutil.copyfile(source_path, destination_path)

    for image in test_images:
        source_path = os.path.join(path, image)
        destination_path = os.path.join(test_class, image)
        # destination_path = test_dir
        shutil.copyfile(source_path, destination_path)

print("Total Training Images: ", total_training_images)
print("Total Validaiton Images: ", total_validation_images)
print("Total Testing Images: ", total_testing_images)