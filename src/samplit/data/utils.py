import os
import subprocess
import tempfile
import time
from contextlib import contextmanager

import pandas as pd
import torch
from tqdm import tqdm

from data.strings import (
  J_CUSTOM_LINES_DIR,
  J_DATAFRAME_CSV,
  J_LYRICS_DIR,
  J_MODEL_LINES,
  J_MP3_DIR,
  J_WORDS_DIR,
  MAX_CHAR_NUM,
)


@contextmanager
def time_it(description: str | None = None):
  """
  context manager used to time pipeline function within terminal exec
  """
  if description:
    tqdm.write(f"{description}")
  start_time = time.time()
  yield
  end_time = time.time()
  execution_time = end_time - start_time
  if execution_time < 60:
    tqdm.write(f"  done! ({execution_time:.4f} sec)\n")
  else:
    tqdm.write(f"  done! ({execution_time / 60:.1f} min)\n")


# jamendo dataset utils
def get_jamendo_track_filepath(track_name: str) -> str:
  """
  get the path to the jamendo track file for the given track name
  """
  return os.path.join(J_MP3_DIR, track_name)


def get_j_word_times_filepath(track_name: str) -> str:
  """
  get the path to the words start and end times file for the given track
  name
  """
  return os.path.join(J_WORDS_DIR, f"{os.path.splitext(track_name)[0]}.csv")


def get_j_words_filepath(track_name: str) -> str:
  """
  get the path to the words file for the given track name
  """
  return os.path.join(J_LYRICS_DIR, f"{os.path.splitext(track_name)[0]}.words.txt")


def get_j_words_df(track_name) -> pd.DataFrame:
  """
  df.columns = ["word_start","word_end","word"]
  """
  words_df = pd.read_csv(get_j_word_times_filepath(track_name))
  words_df = words_df.drop("line_end", axis=1)

  with open(get_j_words_filepath(track_name), "r", encoding="utf-8") as f:
    words: list[str] = f.read().splitlines()

  words_df["word"] = words
  return words_df


def get_j_custom_lines_filepath(
  track_name: str,
  num_lines: int,
  num_words: int,
  min_char_num: int = 1,
  max_char_num: int = MAX_CHAR_NUM,
) -> str:
  """
  get the path to the custom lines file for the given input parameters
  """
  fname_no_ext = os.path.splitext(track_name)[0]
  track_lines_folder = os.path.join(J_CUSTOM_LINES_DIR, fname_no_ext)
  if not os.path.exists(track_lines_folder):
    os.mkdir(track_lines_folder)

  if max_char_num < MAX_CHAR_NUM:
    filename: str = f"{num_words}_words_{num_lines}_lines_{min_char_num}_min_chars_{max_char_num}_max_chars.csv"
  elif min_char_num > 1:
    filename: str = f"{num_words}_words_{num_lines}_lines_{min_char_num}_min_chars.csv"
  else:
    filename: str = f"{num_words}_words_{num_lines}_lines.csv"
  return os.path.join(track_lines_folder, filename)


def get_j_lyrics_filepath(track_name: str) -> str:
  """
  get the path to the lyrics file for the given track name
  """
  return os.path.join(J_LYRICS_DIR, f"{os.path.splitext(track_name)[0]}.txt")


def create_jamendo_dataframe() -> None:
  """
  Merges all dataframes in `J_MODEL_LINES` folders
  """
  dataframes: list[pd.DataFrame] = []
  for track_folder in os.listdir(J_MODEL_LINES):
    folder_path: str = os.path.join(J_MODEL_LINES, track_folder)
    for csv_file in os.listdir(folder_path):
      csv_path: str = os.path.join(folder_path, csv_file)
      cur_df = pd.read_csv(csv_path, index_col=False)
      dataframes.append(cur_df)

  merged_df = pd.concat(dataframes, ignore_index=True)
  merged_df.to_csv(J_DATAFRAME_CSV, index=False)


def load_jamendo_dataframe(csv_file: str = J_DATAFRAME_CSV) -> pd.DataFrame:
  if os.path.exists(csv_file):
    df: pd.DataFrame = pd.read_csv(csv_file)
    df["start_error"] = abs(df["start_time"] - df["model_start_time"])
    df["end_error"] = abs(df["end_time"] - df["model_end_time"])
    df["avg_error"] = (df["start_error"] + df["end_error"]) / 2
    return df


def print_cuda_mem():
  total_memory = torch.cuda.get_device_properties(0).total_memory
  allocated = torch.cuda.memory_allocated(0)
  reserved = torch.cuda.memory_reserved(0)

  tqdm.write(f"  Total: {total_memory / 1024**3:.2f} GB")
  tqdm.write(f"  Allocated: {allocated / 1024**3:.2f} GB")
  tqdm.write(f"  Reserved: {reserved / 1024**3:.2f} GB")


def print_tmp_dir_size():
  if os.name == "posix":
    # os.system(f"du -sh /tmp")
    command = f"du -sh {tempfile.gettempdir()}"
    result = subprocess.run(
      command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    output = result.stdout.strip()
    tqdm.write(output)
  elif os.name == "nt":
    tqdm.write(f"check manually tmp dir size at '{tempfile.gettempdir()}'")