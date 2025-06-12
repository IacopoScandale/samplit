"""
walk in J_CUSTOM_LINES_FOLDER and recreate the same folder structure in
J_MODEL_LINES with the same files but with model answers
"""

import os

# import numpy as np
from spleeter.separator import Separator
from tqdm import tqdm

from data.strings import (
  DEVICE,
  J_CUSTOM_LINES_DIR,
  J_MODEL_LINES,
  SPLEETER_MODEL_PIPELINE,
)
from data.utils import (
  create_jamendo_dataframe,
  get_jamendo_track_filepath,
  print_cuda_mem,
  print_tmp_dir_size,
)
from pipeline_functions import (
  jamendo_model_answer_pipeline,
  load_whisper_model,
  separate_all_jamendo_tracks,
  transcribe_all_jamendo_tracks,
)



# FIXME
if os.name == "nt":
  from phonemizer.backend.espeak.wrapper import EspeakWrapper

  EspeakWrapper.set_library("C:\Program Files\eSpeak NG\libespeak-ng.dll")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# def generate_simplex_grid(step=0.1):
#   grid = []
#   for a in np.arange(0, 1 + step, step):
#     for b in np.arange(0, 1 - a + step, step):
#       c = 1.0 - a - b
#       if c < 0 or c > 1:
#         continue
#       grid.append((a, b, c))
#   return grid


def main() -> None:
  # with tempfile.TemporaryDirectory() as tmp_dir:
  # tmp_dir = None
  print(f"{DEVICE = }")

  # load spleeter model once to avoid tensorflow graph errors
  # embedding model must be loaded every time!
  spleeter_model = Separator(SPLEETER_MODEL_PIPELINE)
  separate_all_jamendo_tracks(spleeter_model)

  # load whisper model once to avoid torch.OutOfMemoryError
  whisper_model = load_whisper_model("large")
  transcribe_all_jamendo_tracks(whisper_model)
  # whisper_model = None
  # bypass: str = "small"

  sim_params: list[float] = [0, 1, 0]

  for track_folder in tqdm(
    os.listdir(J_CUSTOM_LINES_DIR)[40:], desc="Processing Tracks"
  ):
    tqdm.write(f"track: '{track_folder}'")
    # create same folder in model lines if it does not exists
    model_lines_track_folder = os.path.join(J_MODEL_LINES, track_folder)
    if not os.path.exists(model_lines_track_folder):
      os.mkdir(model_lines_track_folder)

    track_folder_path = os.path.join(J_CUSTOM_LINES_DIR, track_folder)

    # not required: monitor tmp dir size because it gets big and it may
    # cause problems
    print_tmp_dir_size()

    if DEVICE == "cuda":
      print_cuda_mem()

    jamendo_model_answer_pipeline(
      get_jamendo_track_filepath(f"{track_folder}.mp3"),
      track_folder_path,
      spleeter_model=spleeter_model,
      whisper_model=whisper_model,
      # _bypass=bypass,
      sim_params=sim_params,
    )

  print("Done!")

  create_jamendo_dataframe()
  print("Created jamendo dataframe!")


if __name__ == "__main__":
  ...
  main()
