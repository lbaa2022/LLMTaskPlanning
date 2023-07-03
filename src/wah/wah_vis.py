import cv2
import os 
from PIL import Image

def create_video(image_paths, output_file):
    # Get image dimensions from the first image in the list
    img = cv2.imread(image_paths[0])
    height, width, _ = img.shape

    # Create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    # Loop over image paths, read images, and write to video
    for path in image_paths:
        img = cv2.imread(path)
        video.write(img)

    # Release video writer and close all windows
    video.release()
    cv2.destroyAllWindows()

for i in [22, 44, 60]:
    for view in ['back', 'front', 'first']:
        fig_dir = f"/home/user/Projects/llm_plan/LMTaskPlanning/figs/230412_video/{view}/{i}"
        x = os.listdir(fig_dir)
        x.sort()
        # img_files = x
        img_files = [os.path.join(fig_dir, file_name) for file_name in x]
        frames = []
        for file in img_files:
            image = Image.open(file)
            frames.append(image)
        frames[0].save(f"{i}_{view}.gif", format="GIF", append_images=frames[1:], save_all=True, duration=200, loop=0)
        create_video(img_files, f"{i}_{view}.mp4")
# pdb.set_trace()