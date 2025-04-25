"""

"""

from data.strings import *
from data.utils import (
  get_j_words_df,
  get_j_lyrics_filepath,
  get_j_custom_lines_filepath,
)
import sys
import random
from tqdm import tqdm
import pandas as pd


def generate_j_unique_lines_df(
  track_name: str,
  num_lines: int, 
  num_words: int, 
  min_char_num: int = 1,
  max_char_num: int = MAX_CHAR_NUM,
) -> pd.DataFrame:
  
  """
  df.columns = ["start_time", "end_time", "lyrics_line"]

  output number of rows is <= num_lines

  for now start with unique lines i.e. lines that appear only once in 
  the full lyrics

  uses `sys.exit()` when there are no lines to extract
  """
  metadata: pd.DataFrame = pd.read_csv(J_LYRICS_CSV)
  track_language: str = metadata[metadata["Filepath"]==track_name]["Language"].iloc[0]

  words_df = get_j_words_df(track_name)
  num_lyrics: int = len(words_df)
  with open(get_j_lyrics_filepath(track_name), "r", encoding="utf-8") as f:
    lyrics: str = f.read()

  start_times: list[float] = [None]*num_lines
  end_times: list[float] = [None]*num_lines
  lyrics_lines: list[str] = [None]*num_lines

  # generate a random start index for a line:
  allowed_start_idxs: list[int] = list(range(num_lyrics-num_words))
  if not allowed_start_idxs:
    print(f"There are no unique lines with {num_words} words")
    sys.exit()
  
  # for i in tqdm(range(num_lines), desc="Finding Lines"):
  for i in range(num_lines):
    early_stop: bool = not allowed_start_idxs
    # like while True, but it stops also with less lines than 'num_lines'
    while allowed_start_idxs:
      start_idx = random.choice(allowed_start_idxs)
      allowed_start_idxs.remove(start_idx)
      end_idx = start_idx + num_words - 1
      
      # check if line is unique
      line: str = " ".join(words_df.iloc[start_idx:end_idx+1]["word"].tolist())

      # check word len
      if len(line) < min_char_num or len(line) > max_char_num:
        continue
      # print(len(line))
        
      # accept only unique lines
      if lyrics.count(line) == 1:
        break

    if not early_stop:
      start_times[i] = words_df.iloc[start_idx]["word_start"]
      end_times[i] = words_df.iloc[end_idx]["word_end"]
      lyrics_lines[i] = line

  lines_df = pd.DataFrame({
    "start_time": start_times,
    "end_time": end_times,
    "lyrics_line": lyrics_lines,
  })
  # export csv
  # lines_df = lines_df.drop_duplicates()
  lines_df = lines_df.dropna()
  lines_df["track_name"] = track_name
  lines_df["num_words"] = num_words
  lines_df["min_char_num"] = min_char_num if min_char_num != 1 else None
  lines_df["max_char_num"] = max_char_num if max_char_num != MAX_CHAR_NUM else None
  lines_df["language"] = track_language

  lines_df.to_csv(
    get_j_custom_lines_filepath(
      track_name, 
      len(lines_df), 
      num_words, 
      min_char_num, max_char_num
    ),
    index=False,
  )
  return lines_df


def main() -> None:
  # (num_lines, num_words, min_char_num, max_char_num)
  choices_for_every_track = [
    (40, 10, 1, MAX_CHAR_NUM),
    (40, 9, 1, MAX_CHAR_NUM),
    (40, 8, 1, MAX_CHAR_NUM),
    (40, 7, 1, MAX_CHAR_NUM),
    (40, 6, 1, MAX_CHAR_NUM),
    (40, 5, 1, MAX_CHAR_NUM),
    (40, 4, 1, MAX_CHAR_NUM),
    (40, 3, 1, MAX_CHAR_NUM),
    (40, 2, 1, MAX_CHAR_NUM),
    (40, 1, 1, MAX_CHAR_NUM),

    (40, 2, 1, 10),
    (40, 2, 10, MAX_CHAR_NUM),

    (40, 1, 1, 5),
    (40, 1, 5, 10),
  ]

  # create custom slices for every track
  for track_name in tqdm(os.listdir(J_MP3_FOLDER), desc="Processing mp3 files"):
    
    # for args in tqdm(choices_for_every_track, desc=f'Processing track "{track_name}"'):
    for args in choices_for_every_track:
      generate_j_unique_lines_df(track_name, *args)


if __name__ == "__main__":
  main()