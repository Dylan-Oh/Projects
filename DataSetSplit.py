import os
import shutil

dataset_path = 'preprocessingData'
labels_path = os.path.join(dataset_path, 'labelsData')
images_path = os.path.join(dataset_path, 'videosFrameData')

os.makedirs('datasets/SMD_Plus', exist_ok=True)
os.makedirs('datasets/SMD_Plus/train', exist_ok=True)
os.makedirs('datasets/SMD_Plus/val', exist_ok=True)

train_dataset_list = ['MVI_1451', 'MVI_1452', 'MVI_1470', 'MVI_1471', 'MVI_1478', 'MVI_1479', 'MVI_1481', 
                      'MVI_1482', 'MVI_1484', 'MVI_1485', 'MVI_1486', 'MVI_1582', 'MVI_1583', 'MVI_1584',
                      'MVI_1609', 'MVI_1610', 'MVI_1612', 'MVI_1622', 'MVI_1623', 'MVI_1624', 'MVI_1627',
                      'MVI_0788', 'MVI_0789', 'MVI_0790', 'MVI_0792', 'MVI_0794', 'MVI_0795', 'MVI_0796',
                      'MVI_0797', 'MVI_0801']
val_dataset_list = ['MVI_1469', 'MVI_1474', 'MVI_1587', 'MVI_1592', 'MVI_1613', 'MVI_1614', 'MVI_1615',
                    'MVI_1644', 'MVI_1645', 'MVI_1646', 'MVI_1448', 'MVI_1640', 'MVI_0799', 'MVI_0804']


for image_dir in os.listdir(images_path):
    for image_file in os.listdir(os.path.join(images_path, image_dir)):
        image_name = image_file.split('_frame')[0]
        
        if image_name in train_dataset_list:
            shutil.copy(os.path.join(images_path, image_dir, image_file),'datasets/SMD_Plus/train')
        elif image_name in val_dataset_list:
            shutil.copy(os.path.join(images_path, image_dir, image_file),'datasets/SMD_Plus/val')
        
for label_dir in os.listdir(labels_path):
    for label_file in os.listdir(os.path.join(labels_path, label_dir)):
        label_name = label_file.split('_frame')[0]
            
        if label_name in train_dataset_list:
            shutil.copy(os.path.join(labels_path, label_dir, label_file),'datasets/SMD_Plus/train')
        elif label_name in val_dataset_list:
            shutil.copy(os.path.join(labels_path, label_dir, label_file),'datasets/SMD_Plus/val')