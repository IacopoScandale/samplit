"""
This script was used to recreate the same folder structure of custom 
JamendoLyrics lines contained in `J_CUSTOM_LINES_FOLDER` directory 
(ground truth data), but also adding the pipeline answers (predicted 
start and end times and predicted custom line transcription) into the 
`J_MODEL_LINES` folder.
"""
import os

from spleeter.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from tqdm import tqdm
from whisper import Whisper

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
  clear_model_lines_dir,
  create_jamendo_dataframe,
  generate_grid,
  get_jamendo_track_filepath,
  get_language_tracks,
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
  spleeter_model: Separator,
  whisper_model: Whisper,
  # spleeter_model_name: str = SPLEETER_MODEL_PIPELINE,
  # whisper_model_name: str = WHISPER_MODEL,
  sim_params: tuple[float, float, float] = SIM_PARAMS,
  language: str | None = None,
) -> None:
  """
  walk in J_CUSTOM_LINES_FOLDER and recreate the same folder structure in
  J_MODEL_LINES with the same files but with model answers
  """
  print(f"{DEVICE = }")

  if language:
    language_tracks: set[str] = get_language_tracks(language)
    tracks: list[str] = [
      track
      for track in os.listdir(J_CUSTOM_LINES_DIR)
      if f"{track}.mp3" in language_tracks
    ]
  else:
    tracks: list[str] = os.listdir(J_CUSTOM_LINES_DIR)

  # print(tracks)
  # input()

  for track_folder in tqdm(tracks[:], desc="Processing Tracks"):
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


def main() -> None:
  # load spleeter model once to avoid tensorflow graph errors
  # obs: embedding model must be loaded every time!
  tqdm.write(f"Loading Spleeter Model '{SPLEETER_MODEL_PIPELINE}'")
  spleeter_model = Separator(SPLEETER_MODEL_PIPELINE)
  separate_all_jamendo_tracks(spleeter_model)

  # load whisper model once to avoid torch.OutOfMemoryError
  tqdm.write(f"Loading Whisper Model '{WHISPER_MODEL}'")
  whisper_model = load_whisper_model(WHISPER_MODEL)
  transcribe_all_jamendo_tracks(whisper_model)
  # whisper_model = None
  # bypass: str = "large"

  for sim_params in tqdm(generate_grid(), desc="Grid Search"):
    out_path: str = os.path.join(
      J_EVALUATION_DIR, "grid_{:.2f}_{:.2f}_{:.2f}.csv".format(*sim_params)
    )
    if os.path.exists(out_path):
      continue

    clear_model_lines_dir()

    tqdm.write(f"{sim_params = }")
    create_model_answers(
      spleeter_model,
      whisper_model,
      sim_params=sim_params, 
      language="English",
    )

    create_jamendo_dataframe(out_path)


if __name__ == "__main__":
  main()
