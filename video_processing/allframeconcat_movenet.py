import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
import numpy as np
import cv2
import skvideo.io
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
import pandas as pd
from moviepy.editor import *
from IPython.display import HTML, display
import time
import os


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2, 
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    """
    Returns high confidence keypoints and edges for visualization

    Parameters
    ----------
    keypoints_with_scores: np.ndarray
        - numpy array of shape (1, 1, 17, 3) representing the keypoint coordinates and scores returned from the MoveNet model.
    height: int
        - height of image in pixels
    width: int
        - width of image in pixels
    keypoint_threshold: float
        - minimum confidence score for a keypoint to be visualized

    Returns
    -------
    keypoints_xy:
        - list of all keypoints of all detected entities
    edges_xy:
        - coordinates of all skeleton edges of all detected entities
    edge_colors:
        - colors in which edges should be plotted
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """
  Draws the keypoint predictions on image.

  Parameters
  ----------
  image: np.ndarray 
    - A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
  keypoints_with_scores: np.ndarray
    - A numpy array with shape [1, 1, 17, 3] representing the keypoint coordinates and scores returned from the MoveNet model.
  crop_region: dict() 
    - A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
  output_image_height: int
    - An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns
  -------
  image_from_plot: np.ndarray
    - A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

# def to_gif(images, fps):
#   """
#   Converts image sequence (4D numpy array) to gif.

#   Parameters
#   ----------
#   images:
#     - collection of images to add to the gif

#   fps:
#     - frame rate of the desired gif

#   Returns
#   -------
#   saved gif file
#   """
#   imageio.mimsave('./animation.gif', images, fps=fps)
#   return embed.embed_file('./animation.gif')

# def progress(value, max=100):
#   return HTML("""
#       <progress
#           value='{value}'
#           max='{max}',
#           style='width: 100%'
#       >
#           {value}
#       </progress>
#   """.format(value=value, max=max))

def load_model(url):
    """
    Load model from tensorflow hub

    Parameters
    ----------
    url: str
        - url of the model

    Returns
    -------
    The desired model
    """
    module = hub.load(url)
    return module

def movenet(module, input_image):
    """
    Runs pose detection on an input image

    Parameters
    ----------
    module:
        - model loaded in from tensorflow_hub
    input_image: tf.tensor
        - input image of shape (1, h, w, 3) that represents input image pixels. Height and width of the image should already be resized to match the expected input resolution of the model before passing into this function

    Returns
    -------
    keypoints_with_scores: np.ndarray
        - numpy array of size (1, 1, 17, 3) representing keypoint coordinates and their confidence scores
    """
    model = module.signatures['serving_default']

    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def downsample_video(video: str, num_frames: int):
    """
    Downsamples a given video clip to a given number of frames.
    Returns an generator object to each frame in the video
    Parameters
    ----------
    video:
        - filename of video without the .mp4 extension
    num_frames:
        - the desired number of frames to extract
    output_file:
        - the filename to put the downsampled video in, with the mp4 extension omitted.
    """
    clip = VideoFileClip(video)
    duration = clip.duration
    # clip_time = min(duration, 1.8)
    clip_time = 1.5
    video_clipped = clip.subclip(0, clip_time)
    video_downsampled = video_clipped.set_fps(num_frames/clip_time)
    return video_downsampled.iter_frames()
    # video_downsampled.write_videofile(f"{output_file}.mp4")
    # return 




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file name of the video (no mp4 suffix)")
    parser.add_argument("input_size", type=int, help="input size of the image")
    parser.add_argument("output_file", type=str, help="name of output file (no csv suffix)")

    args = parser.parse_args()
    file_name = args.file
    input_size = int(args.input_size)
    output_file = args.output_file

    makeFiles = os.listdir(f"{file_name}/makes")
    makeFiles = [f"{file_name}/makes/"+name for name in makeFiles]

    missFiles = os.listdir(f"{file_name}/misses")
    missFiles = [f"{file_name}/misses/"+name for name in missFiles]

    allFiles = makeFiles + missFiles

    for fname in allFiles:
        start = time.time()
        videogen = downsample_video(fname, 60)
        model = load_model("./saved_model")
        row = np.asarray([])
        for frame in videogen:
            image = frame
            input_image = tf.expand_dims(image, axis = 0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
            keypoints_and_scores = movenet(model, input_image)
            keypoints = [x for i,x in enumerate(keypoints_and_scores.flatten()) if i%3!=2 or i==0]
            row = np.append(row, keypoints)
        series = pd.Series(row)
        if('misses' in fname):
            #concat a 0 to the start of the array
            series = pd.concat((pd.Series(0),series))
        elif('makes' in fname):
            # concat a 1 to the start of the array
            series = pd.concat((pd.Series(1),series))

        df = pd.DataFrame([series.tolist()], columns=series.index)
        df.to_csv(f"{output_file}.csv",mode='a+',index=False, header=False)
        end = time.time()
        print("length of row: "+str(len(series.tolist())))
        print("time elapsed (s): "+str(end-start))

if __name__ == '__main__':
    main()
    