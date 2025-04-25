from data.strings import *
import os
import time
from contextlib import contextmanager
import pandas as pd
from tqdm import tqdm

# only to time pipeline functions within terminal
@contextmanager
def time_it(description: str | None = None):
  if description:
    tqdm.write(f"{description}")
  start_time = time.time()  # Start timer
  yield  # Qui viene eseguito il codice all'interno del blocco `with`
  end_time = time.time()  # End timer
  execution_time = end_time - start_time
  if execution_time < 60:
    tqdm.write(f"  done! ({execution_time:.4f} sec)\n")
  else:
    tqdm.write(f"  done! ({execution_time/60:.1f} min)\n")


# jamendo dataset utils
def get_jamendo_track_filepath(track_name: str) -> str:
  """
  get the path to the jamendo track file for the given track name
  """
  return os.path.join(J_MP3_FOLDER, track_name)

def get_j_word_times_filepath(track_name: str) -> str:
  """
  get the path to the words start and end times file for the given track
  name
  """
  return os.path.join(J_WORDS_FOLDER, f"{os.path.splitext(track_name)[0]}.csv")

def get_j_words_filepath(track_name: str) -> str:
  """
  get the path to the words file for the given track name
  """
  return os.path.join(J_LYRICS_FOLDER, f"{os.path.splitext(track_name)[0]}.words.txt")

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
  track_lines_folder = os.path.join(J_CUSTOM_LINES_FOLDER, fname_no_ext)
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
  return os.path.join(J_LYRICS_FOLDER, f"{os.path.splitext(track_name)[0]}.txt")

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

def load_jamendo_dataframe() -> pd.DataFrame:
  if os.path.exists(J_DATAFRAME_CSV):
    df: pd.DataFrame = pd.read_csv(J_DATAFRAME_CSV)
    df["start_error"] = abs(df["start_time"] - df["model_start_time"])
    df["end_error"] = abs(df["end_time"] - df["model_end_time"])
    df["avg_error"] = (df["start_error"] + df["end_error"]) / 2
    return df