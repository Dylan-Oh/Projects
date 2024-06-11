import cv2
import os
import glob

train_data_dir = 'datasets/SMD_Plus/train'
empty_seas_dir = 'emptyseas'

def find_files_with_object_index(directory, object_index):
    # Get list of txt files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    # Initialize a list to store the names of txt files with the specified object index
    files_with_object_index = []

    # Iterate over each txt file
    for txt_file in txt_files:
        # Open the file and read the lines
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Check if any line starts with the specified object index
        for line in lines:
            if line.startswith(str(object_index)):
                # If a line starts with the object index, add the file to the list and break the loop
                files_with_object_index.append(txt_file)
                break

    # Return the list of txt files with the specified object index
    return files_with_object_index


def copy_paste_augmentation(num_augmentations, target_class):
    files_with_object_index = find_files_with_object_index(train_data_dir, target_class)
    for sea_index in range(num_augmentations):
        for file_index in range(len(files_with_object_index)):
            sea_image_name = os.path.join(empty_seas_dir, str(sea_index) + '.jpg')
            target_image_name = files_with_object_index[file_index].replace(".txt", ".jpg")
            target_label_name = files_with_object_index[file_index]
            
            sea_image = cv2.imread(sea_image_name)
            target_image = cv2.imread(target_image_name)
            
            with open(target_label_name, "r") as f:
                target_all_label = f.readlines()
            
            target_label = []
            for line in target_all_label:
                if line.startswith(str(target_class)):
                    # If a line starts with the object index, add the file to the list and break the loop
                    target_label.append(line)
                    break

            if len(target_all_label) != 0:
                for label_index in range(len(target_all_label)):
                    # Parse the label
                    _, x_center, y_center, width, height = map(float, target_label[0].split())
                    
                    # Convert the YOLO coordinates to pixel coordinates
                    img_height, img_width, _ = target_image.shape
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
    
                    # Calculate the start and end coordinates of the bounding box
                    start_x = int(x_center - width / 2)
                    start_y = int(y_center - height / 2)
                    end_x = int(x_center + width / 2)
                    end_y = int(y_center + height / 2)
    
                    # Copy the region from img1
                    region = target_image[start_y:end_y, start_x:end_x]
    
                    # Paste the region into img2
                    sea_image[start_y:end_y, start_x:end_x] = region
                    
                cv2.imwrite(os.path.join(train_data_dir, f"augmented_class_{sea_index}_{target_class}_{file_index}.jpg"), sea_image)
                print(target_label)
                with open(os.path.join(train_data_dir, f"augmented_class_{sea_index}_{target_class}_{file_index}.txt"), "w") as f:
                    f.writelines(target_label)

copy_paste_augmentation(11, 0)

copy_paste_augmentation(17, 1)

copy_paste_augmentation(2, 3)

copy_paste_augmentation(17, 4)

copy_paste_augmentation(17, 5)

copy_paste_augmentation(6, 6)