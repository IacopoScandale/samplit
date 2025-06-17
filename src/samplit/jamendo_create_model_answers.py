import os

import numpy as np
from spleeter.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from tqdm import tqdm

from data.strings import (
  DEVICE,
  J_CUSTOM_LINES_DIR,
  J_EVALUATION_DIR,
  J_MODEL_LINES,
  SIM_PARAMS,
  SPLEETER_MODEL_PIPELINE,
  ESPEAK_NG_DLL,
  WHISPER_MODEL,
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


# espeak-ng windows: https://bootphon.github.io/phonemizer/install.html
if os.name == "nt":
  EspeakWrapper.set_library(ESPEAK_NG_DLL)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_model_answers(
  spleeter_model_name: str = SPLEETER_MODEL_PIPELINE,
  whisper_model_name: str = WHISPER_MODEL,
  sim_params: tuple[float, float, float] = SIM_PARAMS,
) -> None:
  """
  walk in J_CUSTOM_LINES_FOLDER and recreate the same folder structure in
  J_MODEL_LINES with the same files but with model answers
  """
  print(f"{DEVICE = }")

  # load spleeter model once to avoid tensorflow graph errors
  # embedding model must be loaded every time!
  spleeter_model = Separator(spleeter_model_name)
  separate_all_jamendo_tracks(spleeter_model)

  # load whisper model once to avoid torch.OutOfMemoryError
  whisper_model = load_whisper_model(whisper_model_name)
  transcribe_all_jamendo_tracks(whisper_model)
  # whisper_model = None
  # bypass: str = "small"

  for track_folder in tqdm(
    os.listdir(J_CUSTOM_LINES_DIR)[:], desc="Processing Tracks"
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

  tqdm.write("Done!")
  

def generate_grid() -> list[tuple[float, float, float]]:
  """
  weights legend:
  --------------
  1. embedding similarity
  2. phonetic similarity
  3. whisper word probability

  returns:
  -------
  ```
  [(0.0, 0.85, 0.15),
   (0.1, 0.75, 0.15),
   (0.2, 0.65, 0.15),
   (0.3, 0.55, 0.15),
   (0.4, 0.45, 0.15),
   (0.5, 0.35, 0.15),
   (0.6, 0.25, 0.15),
   (0.7, 0.15, 0.15),
   (0.8, 0.05, 0.15)]
   ```
  """
  third: float = 0.15
  step: float = 0.1
  grid: list[tuple[float,float,float]] = []
  for first in np.arange(0.0, 0.8 + step, step):
    first = round(first, 2)
    second = round(1 - (first + third), 2)
    grid.append((first, second, third))
  return grid


def main() -> None:
  for sim_params in generate_grid():
    if os.name == "posix":
      os.system("rm -rf datasets/jamendo_dataset/model_lines/*")
    elif os.name == "nt":
      os.system("del /q /f datasets\jamendo_dataset\model_lines\*")
    # else:
    #   for track_folder in os.listdir(J_MODEL_LINES):
    #     folder_path: str = os.path.join(J_MODEL_LINES, track_folder)
    #     for csv_file in os.listdir(folder_path):
    #       csv_path: str = os.path.join(folder_path, csv_file)
    #       os.remove(csv_path)
    #     os.rmdir(folder_path)

    tqdm.write(f"{sim_params = }")
    create_model_answers(sim_params=sim_params)
    
    out_path: str = os.path.join(
      J_EVALUATION_DIR, 
      "grid_{}_{}_{}.csv".format(*sim_params)
    )
    create_jamendo_dataframe(out_path)


if __name__ == "__main__":
  main()
