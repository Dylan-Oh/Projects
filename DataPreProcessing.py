import cv2
import os
import scipy.io

videosFrameFolder = 'preprocessingData/videosFrameData'
labelsDataFolder = 'preprocessingData/labelsData'

os.makedirs('preprocessingData', exist_ok=True)

image_width = 1920
image_height = 1080

def video_to_frames(video_path, output_path, source_file_name):
    fileID = source_file_name[:8]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        image_file_name = fileID + '_frame%d.jpg' % count
        cv2.imwrite(os.path.join(output_path, image_file_name), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        
def matlab_to_txt(matlab_path, output_path, source_file_name):
    fileID = source_file_name[:8]
    data = scipy.io.loadmat(matlab_path)
    for i in range(data['structXML'].shape[1]):
        label_file_name = fileID + '_frame{:d}.txt'
        filename = os.path.join(output_path, label_file_name.format(i))
        with open(filename, 'w') as f:
            if data['structXML'][0,i]['BB'].size > 0:
                for j in range(data['structXML'][0,i]['BB'].shape[0]):
                    x_center = (data['structXML'][0,i]['BB'][j,0] + (data['structXML'][0,i]['BB'][j,2] / 2)) / image_width
                    y_center = (data['structXML'][0,i]['BB'][j,1] + (data['structXML'][0,i]['BB'][j,3] / 2)) / image_height
                    width = (data['structXML'][0,i]['BB'][j,2]) / image_width
                    height = (data['structXML'][0,i]['BB'][j,3]) / image_height
                    class_index = int(data['structXML'][0,i]['Object'][j]) - 1
                    f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")
                
    

folders = ['VIS_Onboard', 'VIS_Onshore']

for folder in folders:
    video_folder = os.path.join('datasets', folder, 'Videos')
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        video_frame_output_path = os.path.join(videosFrameFolder, os.path.splitext(video_file)[0])
        os.makedirs(video_frame_output_path, exist_ok=True)
        video_to_frames(video_path, video_frame_output_path, video_file)
        
    matlab_folder = os.path.join('datasets', folder, 'ObjectGT')
    for matlab_file in os.listdir(matlab_folder):
        matlab_path = os.path.join(matlab_folder, matlab_file)
        label_data_output_path = os.path.join(labelsDataFolder, os.path.splitext(matlab_file)[0])
        os.makedirs(label_data_output_path, exist_ok=True)
        matlab_to_txt(matlab_path, label_data_output_path, matlab_file)
        