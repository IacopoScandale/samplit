from data.strings import *
import os
import time
from contextlib import contextmanager
import pandas as pd


# only to time pipeline functions within terminal
@contextmanager
def time_it(description: str):
  print(f"\n{description}")
  start_time = time.time()  # Start timer
  yield  # Qui viene eseguito il codice all'interno del blocco `with`
  end_time = time.time()  # End timer
  execution_time = end_time - start_time
  if execution_time < 60:
    print(f"\n  done! ({execution_time:.4f} sec)\n")
  else:
    print(f"\n  done! ({execution_time/60:.1f} min)\n")



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