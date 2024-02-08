import os
import concurrent.futures

# filename= 'aaa.png'
# name_parts = filename.split('_')
# new_name = filename.split('.')[0] + '_cd' + '.png'
# print(name_parts)
# print(new_name)
def rename_file(filename):
    if filename.endswith('.png') or filename.endswith('.tif'):
        # Split the filename and extract the last part before the file extension
        name_parts = filename.split('_')
        # if len(name_parts) == 4:
        #     new_name = name_parts[1] + '_' + name_parts[2]
        # else:
        #     new_name = name_parts[-2]
        # if filename.endswith('_a_pred.png'):
        #     new_name = filename.replace('_a_pred', '')
        #     # new_name = filename.split('.')[0] + '_cd' + '.png'
        # elif filename.endswith('_b_pred.png'):
        #     new_name = filename
        # else:
        #     new_name = filename.split('.')[0] + '_cd' + '.png'
        # new_name = name_parts[-2]
        new_name = filename.replace('test_', '')
        if new_name.endswith('_0.png_0.png'):
            new_name = new_name.replace('_0.png_0.png', '_0.png')
        else:
            new_name = new_name.replace('_0.png', '')
        new_name = new_name.replace('jpg', 'png')
        # new_name = new_name.replace('tif', 'png')
        # Construct the full old and new file paths
        old_file_path = os.path.join(source_dir, filename)
        new_file_path = os.path.join(destination_dir, new_name)
        # Rename the file
        # os.rename(old_file_path, new_file_path)
        os.system(f'sudo cp {old_file_path} {new_file_path}')
        return f'Renamed {filename} to {new_name}'
    return f'Skipped {filename}'

source_dir = '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/unet/vis_data/vis_image'  # Replace with the path to your source directory
destination_dir = '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/unet/vis_data/vis_image_changed'  # Replace with the path to your destination directory

os.system(f'sudo rm -rf {destination_dir}')
# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Get a list of files to be renamed
files_to_rename = [f for f in os.listdir(source_dir) if f.endswith('.png') or f.endswith('.tif')]

# Use a process pool to rename files in parallel
with concurrent.futures.ProcessPoolExecutor(64) as executor:
    results = executor.map(rename_file, files_to_rename)

    # Iterate over the results and print them
    for result in results:
        print(result)