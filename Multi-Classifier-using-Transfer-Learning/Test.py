"""
TEST_DIR = DDP.test_dir
CLASS_LIST = DDP.class_list
TEST_IMAGE_COUNT = 0

for folder in CLASS_LIST:
    # predictions
    print("Predictions for images in " + folder + " class")
    path = os.path.join(TEST_DIR, folder)
    path_list = os.listdir(path)

    for fn in path_list:
        img_path = os.path.join(path, fn)
        try:
            img = image.load_img(img_path, target_size = (300, 300))

            xs = image.img_to_array(img)
            xs = np.expand_dims(xs, axis = 0)

            images = np.vstack([xs])
            classes = model.predict(images, batch_size = 20)
            # print(classes)

            TEST_IMAGE_COUNT += 1
            CLASS_IMAGE_INDEX = 0
            print("Test Image no.: ", TEST_IMAGE_COUNT)
            for idx in range(150):
                if classes[CLASS_IMAGE_INDEX][idx] > 0.6:
                    print("\n" + fn + " is a " + CLASS_LIST[idx])
        
            CLASS_IMAGE_INDEX += 1
        
        except:
            continue
"""


