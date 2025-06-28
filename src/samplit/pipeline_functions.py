"""
This script contains all the samplit pipeline functions to make the 
evaluation scripts and the streamlit application script work.
"""
import json
import os
import re

import ffmpeg
import Levenshtein
import numpy as np
import pandas as pd
import torch
import whisper
from phonemizer import phonemize
from sentence_transformers import SentenceTransformer
from spleeter.separator import Separator
from tqdm import tqdm
from whisper import Whisper

from data.strings import (
  CACHE_HUGGING_FACE,
  DEVICE,
  J_MODEL_LINES,
  J_MP3_DIR,
  JSON_SAVES,
  MODELS_PATH,
  PHONEMIZE_BACKEND,
  SEPARATED_TRACKS_PATH,
  SIM_PARAMS,
  SLICES_FOLDER,
  SPLEETER_MODEL_PIPELINE,
  WHISPER_MODEL,
  WHISPER_TO_ESPEAK_LAN,
)
from data.utils import get_jamendo_track_filepath, time_it


def separate_all_tracks(
  audio_path: str,
  spleeter_model: Separator | None = None,
) -> dict[str, str]:
  """
  Download spleeter model (5 stems) if necessary.

  Create following audio files:

  - `SEPARATED_TRACKS_PATH` (main folder)
      - `audio_path` (folder: track name without extension)
          - "bass.wav"
          - "drums.wav"
          - "other.wav"
          - "piano.wav"
          - "vocals_for_transcription.wav"
          - "vocals.wav"

  "vocals_for_transcription.wav" track is explicitly created from
  vocals, but in mono at 16000 Hz, with s16 format for a better
  transcription

  Parameters
  ----------
  audio_path : str
      Audio track path

  Returns
  -------
  dict[str,str]
      Returns a dict where the key are the following:
      ["original","vocals","vocals for transcription", "piano","drums",
      "bass","other"]
      and values are the relative track paths.
  """
  # get only filename with and witout ext
  filename: str = os.path.basename(audio_path)  # with ext
  fname, ext = os.path.splitext(filename)

  # folder where spleeter puts vocals and accompaniment tracks
  separated_track_folder = os.path.join(SEPARATED_TRACKS_PATH, fname)

  if not os.path.exists(separated_track_folder):
    # load spleeter model if not passed as argument
    if not spleeter_model:
      # spleeter to separate audio voice from instruments
      spleeter_model: Separator = Separator(SPLEETER_MODEL_PIPELINE)
    spleeter_model.separate_to_file(audio_path, SEPARATED_TRACKS_PATH)

  vocals_for_transcription = os.path.join(
    separated_track_folder, "vocals_for_transcription.wav"
  )
  vocals = os.path.join(separated_track_folder, "vocals.wav")
  piano: str = os.path.join(separated_track_folder, "piano.wav")
  drums: str = os.path.join(separated_track_folder, "drums.wav")
  bass: str = os.path.join(separated_track_folder, "bass.wav")
  other: str = os.path.join(separated_track_folder, "other.wav")

  # convert wav to 16000 Hz, mono, s16 format for better transcription
  if not os.path.exists(vocals_for_transcription):
    # os.system(f'ffmpeg -i "{vocals}" -ar 16000 -ac 1 -sample_fmt s16 "{vocals_for_transcription}"')
    (
      ffmpeg
      .input(vocals)
      .output(vocals_for_transcription, ar=16000, ac=1, sample_fmt="s16")
    ).run()

  return {
    "original": audio_path,
    "vocals": vocals,
    "vocals for transcription": vocals_for_transcription,
    "piano": piano,
    "drums": drums,
    "bass": bass,
    "other": other,
  }


def track_name_to_json_transcription(
  filepath: str,
  whisper_model: str = WHISPER_MODEL,
) -> str:
  """
  Parameters
  ----------
  filepath : str
      file path

  whisper_model : str
      e.g. "medium", "large",...

  Returns
  -------
  str
      mapped json filename to save transcription

  Examples
  --------
  >>> track_name_to_json_transcription("folder/file.wav")
      "'JSON_SAVES'/file.json"
  """
  filename = os.path.basename(os.path.dirname(filepath))
  fname, ext = os.path.splitext(filename)
  whisper_model_dir: str = os.path.join(JSON_SAVES, whisper_model)
  if not os.path.exists(whisper_model_dir):
    os.mkdir(whisper_model_dir)
  transcription_json = os.path.join(whisper_model_dir, fname + ".json")
  return transcription_json


def load_whisper_model(model_name: str = WHISPER_MODEL) -> Whisper:
  model: Whisper = whisper.load_model(
    model_name, device=DEVICE, download_root=MODELS_PATH
  )
  # just because it does not fit in my gpu, otherwise autocast does it
  # all automatically
  model = model.half() if DEVICE == "cuda" else model
  model.model_name = model_name
  return model


def transcribe_with_timestamps_whisper(
  file_path: str,
  print_final_result: bool = False,
  whisper_model: Whisper | None = None,
  _bypass: str | None = None,
) -> dict:
  """
  Transcribes a wav `file_path` using Whisper

  Saves the transcription dict as json in `JSON_SAVES` dir.

  Parameters
  ----------

  file_path : str
      wav audio path

  print_final_result : bool, optional
      if true prints final transcription

  whisper_model : str
      e.g. "medium", "large",...

  Returns
  -------
  dict
      Un dizionario contenente la trascrizione completa e i dettagli delle parole con timestamp.

  Examples
  --------
  >>> transcribe_with_timestamps_vosk("path/to/audio.wav")
      {'result': [...], 'text': 'trascrizione completa'}
  """
  # check if transcription already exists in JSON_SAVES and save time
  if _bypass:
    transcription_json = track_name_to_json_transcription(file_path, _bypass)
  elif whisper_model:
    transcription_json = track_name_to_json_transcription(
      file_path, whisper_model.model_name
    )
  else:
    transcription_json = track_name_to_json_transcription(file_path, WHISPER_MODEL)
  if os.path.exists(transcription_json):
    with open(transcription_json, "r", encoding="utf-8") as jsonfile:
      transcribed_dict = json.load(jsonfile)
    return transcribed_dict

  if not whisper_model:
    whisper_model = load_whisper_model()

  with torch.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
    dict_res = whisper_model.transcribe(file_path, word_timestamps=True)

  # duration_tol: float = 0.20
  # dict_res["result"] = [word for segment in dict_res["segments"] for word in segment["words"] if word["end"]-word["start"] >= duration_tol]
  dict_res["result"] = [
    word for segment in dict_res["segments"] for word in segment["words"]
  ]
  dict_res["text"] = " ".join(word_dict["word"] for word_dict in dict_res["result"])

  if print_final_result:
    print(dict_res["text"])

  # save json to JSON_SAVES
  with open(transcription_json, "w", encoding="utf-8") as jsonfile:
    json.dump(dict_res, jsonfile, indent=2)

  return dict_res


def clean_filename(filename: str) -> str:
  """
  Remove following problematic characters <>:"/\\\|?*\\x00 to \\x1F from
  input `filename`

  Parameters
  ----------
  filename : str
      Some string

  Returns
  -------
  str
      Same input string without problematic characters that do not let
      files to be saved with

  Examples
  --------
  >>> clean_filename("/Am I?.mp3")
      "Am I.mp3"
  """
  filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", filename)
  return filename


def k_best_chunks_with_phonetic_embeddings(
  wav_path: str,
  queries: list[str],
  k: int,
  transcribed_dict: dict | None = None,
  # embedding_model: SentenceTransformer | None = None,
  # tmp_dir: str | None = None,
  verbose: bool = True,
  device: str = "cpu",  # faster and does not allocate anything on gpu
  sim_params: list[int | float] = SIM_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Finds the `k` best chunks for each query in `queries` for the audio
  file at `wav_path` and returns two np.ndarray of shape
  `(len(queries), k)` with the start and end times of the respective
  queries

  Parameters
  ----------
  wav_path : str
      Audio file path

  queries : list[str]
      List of queries to find in the audio

  k : int
      Number of top best chunks to export

  transcribed_dict : dict | None = None
      If precalculated accepts transcription dictionary to save time

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
      Returns `(start_times, end_times)` each one of shape
      `(len(queries), k)`

  Examples
  --------
  >>> start_times, end_times = k_best_chunks_with_phonetic_embeddings("audio.wav", ["my love", "you", "hello"], k=2)
  >>> start_times
  array([[10.2, 30.1],  # Start times for "my love"
         [15.0, 45.3],  # Start times for "you"
         [5.5, 40.0]])  # Start times for "hello"
  >>> end_times
  array([[12.5, 32.0],  # End times for "my love"
         [17.2, 47.5],  # End times for "you"
         [7.5, 42.0]])  # End times for "hello"
  """

  # # put espeak-ng tmp files in a temporary directory
  # if tmp_dir:
  #   os.environ["TMPDIR"] = tmp_dir

  # save some time
  if not transcribed_dict:
    transcription_json = track_name_to_json_transcription(wav_path)
    if os.path.exists(transcription_json):
      with open(transcription_json, "r", encoding="utf-8") as jsonfile:
        transcribed_dict: dict = json.load(jsonfile)
    else:
      transcribed_dict: dict = transcribe_with_timestamps_whisper(wav_path)

  whisper_lan: str = transcribed_dict["language"]
  espeak_lan: str = WHISPER_TO_ESPEAK_LAN[whisper_lan]
  if verbose:
    print(f"{whisper_lan = }")
    print(f"{espeak_lan = }")

  # Embedding model: it is not explicitly multilingual, but works fine,
  # it is lightweight and fast and is better than other multilingual
  # pretrained models on hugging face.
  # if not embedding_model:
  embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device,
    cache_folder=CACHE_HUGGING_FACE,
  )

  # transcription = transcribed_dict["text"]
  # transcription_words = transcription.split()
  # cannot use this due to '¿ querés?' considered as a single word by whisper
  transcription_words = [
    word_dict["word"].strip()
    for word_dict in transcribed_dict["result"]
    if word_dict["word"].strip() != ""
  ]

  start_times = np.zeros((len(queries), k))
  end_times = np.zeros((len(queries), k))
  transcriptions = np.full(
    (len(queries), k), "", dtype=object
  )  # object because str has a fixed size

  word_probabilities = np.array(
    [
      word_dict["probability"]
      for word_dict in transcribed_dict["result"]
      if word_dict["word"].strip() != ""
    ]
  )

  # only two calls to phonemize for all needed words
  all_query_unique_words: list[str] = list(
    set(word for query in queries for word in clean_filename(query).split())
  )
  all_phonemes: list[str] = phonemize(
    all_query_unique_words, language=espeak_lan, backend=PHONEMIZE_BACKEND
  )
  query_word_to_phoneme: dict[str, str] = dict(
    zip(all_query_unique_words, all_phonemes)
  )

  all_chunk_unique_words: list[str] = list(set(transcription_words))
  all_chunk_phonemes: list[str] = phonemize(
    all_chunk_unique_words, language=espeak_lan, backend=PHONEMIZE_BACKEND
  )
  chunk_word_to_phoneme: dict[str, str] = dict(
    zip(all_chunk_unique_words, all_chunk_phonemes)
  )
  # case when "" is in the transcription
  # if len(all_chunk_unique_words) > len(all_chunk_phonemes):

  for query_idx, query in tqdm(
    enumerate(queries), desc="Processing Queries", total=len(queries), leave=False
  ):
    query = clean_filename(query)
    query_words = query.split()

    len_transcription, len_query = len(transcription_words), len(query_words)

    contiguous_chunks = [
      transcription_words[i : i + len_query]
      for i in range(len_transcription - len_query + 1)
    ]
    # bad case when there are more words in the query then in the transcription
    if len(contiguous_chunks) == 0:
      # tqdm.write("bad case")
      contiguous_chunks = [transcription_words.copy()]

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    chunks_embeddings = embedding_model.encode(
      [" ".join(chunk) for chunk in contiguous_chunks], convert_to_numpy=True
    )

    similarities = (
      embedding_model.similarity(chunks_embeddings, query_embedding).numpy().reshape(-1)
    )
    # cos_sim = util.cos_sim(chunks_embeddings, query_embedding).numpy()

    phonetic_similarities = np.zeros(len(contiguous_chunks))

    for i, chunk in tqdm(
      enumerate(contiguous_chunks),
      desc="Phonetic similarities",
      total=len(contiguous_chunks),
      disable=not verbose,
      leave=False,
    ):
      cur_sim = np.array(
        [
          Levenshtein.ratio(
            chunk_word_to_phoneme[chk_word], query_word_to_phoneme[query_word]
          )
          for chk_word, query_word in zip(chunk, query_words)
        ]
      )
      phonetic_similarities[i] = np.mean(cur_sim)

    # medie delle probabilità delle parole trascritte per ogni chunk
    kernel = np.ones(len_query) / len_query
    chunk_mean_probabilities = np.convolve(word_probabilities, kernel, mode="valid")
    # bad case if there is only one chunk
    if len(contiguous_chunks) == 1:
      chunk_mean_probabilities = np.array([chunk_mean_probabilities[0]])
    # res_sim = 0.4*similarities + 0.6*phonetic_similarities
    # res_sim = 0.35*similarities + 0.55*phonetic_similarities + 0.1*chunk_mean_probabilities
    # res_sim = 0.35*similarities + 0.50*phonetic_similarities + 0.15*chunk_mean_probabilities
    a, b, c = sim_params
    res_sim = (
      a * similarities + b * phonetic_similarities + c * chunk_mean_probabilities
    )

    top_k_idx = np.argsort(res_sim)[::-1][:k]
    # best_chunk_idx = np.argmax(res_sim)
    # best_chunk_idx = top_k_idx[0]
    # print(f"Best chunk for query '{query}' is:\n  --> {contiguous_chunks[best_chunk_idx]}")
    # update top k transcriptions for current query
    for i, idx in enumerate(top_k_idx):
      transcriptions[query_idx, i] = " ".join(contiguous_chunks[idx])

    if verbose:
      print(f"Top {k} chunks for query '{query}':")
      for i, idx in enumerate(top_k_idx, 1):
        print(
          f"  {i:>2}. {' | '.join(contiguous_chunks[idx]):<70}\t sim: {res_sim[idx]:.5f}\t cos sim: {similarities[idx]:.5f}\t phonetic sim: {phonetic_similarities[idx]:.5f}\t  prob: {chunk_mean_probabilities[idx]:.5f}"
        )
      print()

    for k_idx, idx in enumerate(top_k_idx):
      chunk_dict = transcribed_dict["result"][idx : idx + len_query]
      start_time: float = chunk_dict[0]["start"]
      end_time: float = chunk_dict[-1]["end"]

      start_times[query_idx, k_idx] = start_time
      end_times[query_idx, k_idx] = end_time
      
  # remove tmp dir in env
  # if tmp_dir:
  # os.environ.pop("TMPDIR", None)
  return start_times, end_times, transcriptions


def full_instruments_ffmpeg_cut(
  start_time: float,
  end_time: float,
  track_names: dict[str, str],
  query: str,
  top_k_position: int,
  selections: dict[str, bool],
) -> str:
  """
  Cuts selected start and end times using ffmpeg. Select from one to
  multiple instruments and merges into the final track

  Parameters
  ----------
  start_time : float
      The start time from which the audio should be cut.

  end_time : float
      The end time until which the audio should be cut.

  track_names : dict[str, str]
      A dictionary mapping instrument names (e.g., "vocals", "piano",
      etc.) to their corresponding file paths.

  query : str
      A string used for naming the output file, typically related to the
      search query.

  top_k_position : int
      The ranking position of the track in the search results.

  selections : dict[str, bool]
      A dictionary indicating which instrument tracks should be included
      in the final mix. Keys correspond to instrument names, and values
      indicate selection.

  Returns
  -------
  str
      The path of the extracted audio file
  """
  # if transcription model gives start_time >= end_time there is 
  # something strange, so just swap them and add and subtract epsilon
  if start_time >= end_time:
    start_time, end_time = end_time-0.3, start_time+0.3 

  query = clean_filename(query)
  track_name: str = os.path.basename(os.path.splitext(track_names["original"])[0])

  # oss from selection is not possible to select original with any other
  keys: list[str] = ["original", "vocals", "piano", "drums", "bass", "other"]
  desc: str = "_".join(k for k in keys if selections[k])
  selected_tracks: list[str] = [track_names[k] for k in keys if selections[k]]

  # ffmpeg export
  track_desc: str = (
    f"{track_name}_{'_'.join(query.split())}_{top_k_position}_{desc}.wav"
  )
  out_track_name: str = os.path.join(SLICES_FOLDER, track_desc)

  if len(selected_tracks) == 1:
    stream = (
      ffmpeg
      .input(track_names[desc])
      .output(out_track_name, ss=start_time, to=end_time)
    )

  else:
    inputs = [ffmpeg.input(track) for track in selected_tracks]
    stream = (
      ffmpeg
      .filter(inputs, "amix", inputs=len(selected_tracks), normalize=0)
      .output(out_track_name, ss=start_time, to=end_time)
    )

  if os.path.exists(out_track_name):
    os.remove(out_track_name)

  try:
    stream.run()
  except Exception as e:
    print(repr(e))
    print(f"{start_time = }")
    print(f"{end_time = }") 
    input()

  # st.write(os.path.basename(out_track_name))
  # st.write(f"time interval : [{start_time},{end_time}]")
  # st.audio(out_track_name)
  # print()
  # following code only for streamlit ui
  return out_track_name


def terminal_pipeline(
  track: str,
  # queries: list[str],
  query: str,
  k: int = 3,
):
  with time_it("Audio Separation..."):
    track_names: dict[str, str] = separate_all_tracks(track)

  with time_it("Transcription..."):
    transcribed_dict = transcribe_with_timestamps_whisper(
      track_names["vocals for transcription"]
    )

  with time_it("Chunking..."):
    start_times, end_times, transcriptions = k_best_chunks_with_phonetic_embeddings(
      track_names["vocals"], [query], k, transcribed_dict
    )

  # just for this use extract from original audio track
  selections: dict[str, bool] = {
    "original": True,
    "vocals": False,
    "piano": False,
    "drums": False,
    "bass": False,
    "other": False,
  }

  with time_it("Final ffmpeg cut"):
    top_extracted_cuts: list[str] = [""] * (k)
    for i in range(k):
      top_extracted_cuts[i] = full_instruments_ffmpeg_cut(
        start_times[0, i],
        end_times[0, i],
        track_names,
        query,
        i + 1,
        selections,
      )

  # following only for streamlit ui
  start_end_times = list(zip(start_times[0, :], end_times[0, :]))
  return track_names, top_extracted_cuts, start_end_times, transcriptions[0, :]


def jamendo_model_answer_pipeline(
  track_path: str,
  track_folder_path: str,
  # lines_df: pd.DataFrame,
  # queries: list[str],
  k: int = 1,
  spleeter_model: Separator | None = None,
  # embedding_model: SentenceTransformer | None = None,
  # espeak_tmp_dir: str | None = None,
  sim_params: list[int | float] = SIM_PARAMS,
  whisper_model: Whisper | None = None,
  _bypass: str | None = None,  # to debug
):
  """
  cfr src/samplit/create_jamendo_model_answers.py
  """
  track_name_no_ext = os.path.basename(os.path.splitext(track_path)[0])
  # lines_df = pd.read_csv(os.path.join(lines, f"{track_name_no_ext}.csv"))

  with time_it("Audio Separation..."):
    track_names: dict[str, str] = separate_all_tracks(track_path, spleeter_model)

  with time_it("Transcription..."):
    transcribed_dict = transcribe_with_timestamps_whisper(
      track_names["vocals for transcription"],
      whisper_model=whisper_model,
      _bypass=_bypass,
    )

  tqdm.write("Chunking...")
  for lines_csv in tqdm(
    os.listdir(track_folder_path), desc="Processing Lines", leave=False
  ):
    lines_csv_path = os.path.join(track_folder_path, lines_csv)
    lines_csv_filename = os.path.basename(lines_csv_path)
    lines_df = pd.read_csv(lines_csv_path)
    queries: list[str] = lines_df["lyrics_line"].tolist()

    # with time_it("Chunking..."):
    start_times, end_times, transcriptions = k_best_chunks_with_phonetic_embeddings(
      track_names["vocals"],
      queries,
      k,
      transcribed_dict,
      # embedding_model,
      # tmp_dir=espeak_tmp_dir,
      verbose=False,
      sim_params=sim_params,
    )
    lines_df["model_start_time"] = start_times[:, 0]
    lines_df["model_end_time"] = end_times[:, 0]
    lines_df["model_transcription"] = transcriptions[:, 0]
    # out_csv = os.path.join(model_lines,f"{track_name_no_ext}_pipeline.csv")
    out_csv: str = os.path.join(J_MODEL_LINES, track_name_no_ext, lines_csv_filename)
    lines_df.to_csv(out_csv, index=False)


def separate_all_jamendo_tracks(spleeter_model: Separator) -> None:
  """
  Run spleeter to every jamendo dataset track to avoid device-related
  pipeline errors
  """
  for track in tqdm(os.listdir(J_MP3_DIR), desc="Splitting Tracks"):
    track_path: str = get_jamendo_track_filepath(track)
    separate_all_tracks(track_path, spleeter_model)


def transcribe_all_jamendo_tracks(whisper_model: Whisper) -> None:
  """
  Run whisper after separation to every jamendo dataset track to avoid
  device-related pipeline errors
  """
  for track in tqdm(
    os.listdir(J_MP3_DIR), desc=f"Whisper Transcription '{whisper_model.model_name}'"
  ):
    track_path: str = get_jamendo_track_filepath(track)
    track_names: dict[str, str] = separate_all_tracks(track_path)
    transcribe_with_timestamps_whisper(
      track_names["vocals for transcription"],
      whisper_model=whisper_model,
    )