"""
walk in J_CUSTOM_LINES_FOLDER and recreate the same folder structure in
J_MODEL_LINES with the same files but with model answers
"""

from data.strings import *
from data.utils import get_jamendo_track_filepath, create_jamendo_dataframe
from pipeline_functions import jamendo_model_answer_pipeline
import os
import tempfile
import subprocess
from tqdm import tqdm
from spleeter.separator import Separator


# FIXME
if os.name == "nt":
  from phonemizer.backend.espeak.wrapper import EspeakWrapper
  EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main() -> None:
  # with tempfile.TemporaryDirectory() as tmp_dir:
  # tmp_dir = None
  print(f"{DEVICE = }")

  # load spleeter model once to avoid tensorflow graph errors
  spleeter_model = Separator(SPLEETER_MODEL)
  # embedding model must be loaded every time!

  for track_folder in tqdm(os.listdir(J_CUSTOM_LINES_FOLDER)[40:], desc="Processing Tracks"):
    # create same folder in model lines if it does not exists
    model_lines_track_folder = os.path.join(J_MODEL_LINES, track_folder)
    if not os.path.exists(model_lines_track_folder):
      os.mkdir(model_lines_track_folder)

    track_folder_path = os.path.join(J_CUSTOM_LINES_FOLDER, track_folder)
  
    # not required: monitor tmp dir size because it gets big and it may
    # may cause problems
    if os.name == "posix":
      # os.system(f"du -sh /tmp")
      command = f"du -sh {tempfile.gettempdir()}"
      result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
      output = result.stdout.strip()
      tqdm.write(output)


    jamendo_model_answer_pipeline(
      get_jamendo_track_filepath(f"{track_folder}.mp3"),
      track_folder_path,
      spleeter_model=spleeter_model,
    )



if __name__ == "__main__":
  main()
  print("Done!")
  create_jamendo_dataframe()
  print("Created jamendo dataframe!")