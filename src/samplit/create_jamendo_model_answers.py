"""
walk in J_CUSTOM_LINES_FOLDER and recreates the same folder structure in
J_MODEL_LINES with the same files but with model answers
"""

from data.strings import *
from data.utils import get_jamendo_track_filepath
from pipeline_functions import jamendo_model_answer_pipeline
import os
from tqdm import tqdm


for track_folder in tqdm(os.listdir(J_CUSTOM_LINES_FOLDER), desc="Processing Tracks"):
  # create same folder in model lines if it does not exists
  model_lines_track_folder = os.path.join(J_MODEL_LINES, track_folder)
  if not os.path.exists(model_lines_track_folder):
    os.mkdir(model_lines_track_folder)

  track_folder_path = os.path.join(J_CUSTOM_LINES_FOLDER, track_folder)
  for lines_csv in tqdm(os.listdir(track_folder_path), desc="Processing Lines", leave=False, position=1):
    lines_csv_path = os.path.join(track_folder_path, lines_csv)

    jamendo_model_answer_pipeline(
      get_jamendo_track_filepath(f"{track_folder}.mp3"),
      lines_csv_path,
    )

