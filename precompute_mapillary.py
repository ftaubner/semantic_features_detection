import cv2
import imageio
import os
import time
import numpy as np
import glob
import threading


def compute_classes(data_path, file_name):
    """Precompute image to class color map for every file.
    """
    instance_masks = []
    class_colors = []

    # tic = time.perf_counter()

    instance_path = os.path.join(data_path, 'instances', file_name)
    class_path = os.path.join(data_path, 'labels', file_name)

    instance_im = imageio.imread(instance_path)
    label_im = imageio.imread(class_path)

    # Find unique instances in image and map their class color.
    instance_ids = np.unique(instance_im)
    for instance_id in instance_ids:
        first_where_index = np.unravel_index(np.argmax(instance_im == instance_id), instance_im.shape)
        class_color = label_im[first_where_index[0], first_where_index[1], :3]
        class_colors.append(class_color)

    # Write instances and colors to file.
    with open("{}.txt".format(instance_path[:-3]), 'w') as f:
        for i in range(len(instance_ids)):
            f.write("{} {} {} {}\n".format(instance_ids[i], class_colors[i][0], class_colors[i][1], class_colors[i][2]))

    # toc = time.perf_counter()
    # print("Time to stack masks instances: {}".format(toc-tic))


def process_files(data_path, file_list, start_index, file_count, worker_idx):
    print("Starting worker {}.".format(worker_idx))
    tic = time.perf_counter()

    for i in range(start_index, start_index + file_count):
        file_name = file_list[i].split('/')[-1]
        compute_classes(data_path, file_name)
        if (i + 1) % 10 == 0:
            print("Worker {} working on image {}/{}".format(worker_idx, i + 1 - start_index, file_count))

    d_t = time.perf_counter() - tic
    print("Worker {} finished.".format(worker_idx))
    print("Precomputed {} images in {} seconds.".format(file_count, d_t))
    print("Average time per image: {}s.".format(d_t / file_count))


if __name__ == '__main__':
    dataset_dir = "/tf/data/mapillary_vistas/"

    num_workers = 8

    # Training
    dir_name = 'training'
    file_list = glob.glob(os.path.join(dataset_dir, dir_name, 'instances', '*.png'))

    print("Precomputing training and validation set, this may take a while.")
    print("Starting {} workers for training set.".format(num_workers))

    file_count = len(file_list)
    data_path = os.path.join(dataset_dir, dir_name)

    for i in range(num_workers):
        thread = threading.Thread(target=process_files,
                                  args=(data_path,
                                        file_list,
                                        int(i * file_count / num_workers),
                                        int(file_count / num_workers) + 1,
                                        i))
        thread.start()

    # Validation.
    dir_name = 'validation'
    file_list = glob.glob(os.path.join(dataset_dir, dir_name, 'instances', '*.png'))

    num_workers = 1

    print("Starting {} workers for validation set.".format(num_workers))

    file_count = len(file_list)
    data_path = os.path.join(dataset_dir, dir_name)

    for i in range(num_workers):
        thread = threading.Thread(target=process_files,
                                  args=(data_path,
                                        file_list,
                                        int(i * file_count / num_workers),
                                        int(file_count / num_workers) + 1,
                                        i))
        thread.start()

    print("Finished precomputing.")



