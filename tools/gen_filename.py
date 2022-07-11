import os

def gen_train_txt(data_dir, interval_list, filename_dir, mode):
    full_data_dir = os.path.join(data_dir, mode)
    scene_list = os.listdir(full_data_dir)
    scene_list.sort()
    filename_dir_ = os.path.join(filename_dir, 'train.txt')
    f = open(filename_dir_,'w')
    for scene in scene_list:
        for interval in interval_list:
            blur_list = os.listdir(os.path.join(full_data_dir, scene, 'blur_images', '10', interval))
            num_blur = len(blur_list)
            interval0 = int(interval.split('-')[0])
            interval1 = int(interval.split('-')[1])
            for blur_idx in range(num_blur-1):
                f.write(scene + ' ' + str(interval0+interval1) + ' ' +  interval + ' ' + \
                        str(blur_idx).zfill(5) + ' ' + str(blur_idx+1).zfill(5) + '\n')

def gen_test_txt(data_dir, interval_list, filename_dir, mode):
    full_data_dir = os.path.join(data_dir, 'test')
    for interval in interval_list:
        filename_dir_ = os.path.join(filename_dir, 'test_' + interval + '.txt')
        f = open(filename_dir_,'w')
        scene_list = os.listdir(full_data_dir)
        for scene in scene_list:
            blur_list = os.listdir(os.path.join(full_data_dir, scene, 'blur_images', '14', interval))
            num_blur = len(blur_list)
            interval0 = int(interval.split('-')[0])
            interval1 = int(interval.split('-')[1])
            for blur_idx in range(num_blur - 1):
                f.write(scene + ' ' + str(interval0+interval1) + ' ' +  interval + ' ' \
                        + str(blur_idx).zfill(5) + ' ' + str(blur_idx+1).zfill(5) + '\n')


if __name__=='__main__':
    data_dir = '/home/user/intelpro/dataset/dataset/unknown_exposure_public/'
    filename_dir = '../filename/dvs_dataset'
    gen_train_txt(data_dir, ['3-7', '5-5', '7-3', '9-1'], filename_dir, 'train')
    filename_dir = '../filename/dvs_dataset'
    gen_test_txt(data_dir, ['9-5', '11-3', '13-1'], filename_dir, 'train')
    # gen_train_txt(data_dir, interval_list=['9-5', '11-3', '13-1'])