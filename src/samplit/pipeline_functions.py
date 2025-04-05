from data.strings import *
# from .data.utils import time_it
import os
import re
import json
from tqdm import tqdm
import ffmpeg
import numpy as np
from sentence_transformers import SentenceTransformer
from spleeter.separator import Separator
import whisper
from phonemizer import phonemize
import Levenshtein
# import streamlit as st


def separate_all_tracks(audio_path: str) -> dict[str,str]:
  """
  Scarica il modello di spleeter (5 stems) se necessario.
  
  Crea i file audio di voce e parte strumentale con la seguente 
  gerarchia:

  - SEPARATED_TRACKS_PATH (main folder)
      - `audio_path` (folder: track name without extension)
          - "bass.wav"
          - "drums.wav"
          - "other.wav"
          - "piano.wav"
          - "vocals_for_transcription.wav"
          - "vocals.wav"
  
  Inoltre crea anche "vocals_for_transcription.wav", ovvero la stessa
  traccia di vocals ma convertita a 16000 Hz, mono con formato s16 per
  una migliore trascrizione
          
  Parameters
  ----------
  audio_path : str
      Audio track path
  
  Returns
  -------
  dict[str,str]
      Restituisce un dizionario con i path delle tracce avente le 
      seguenti chiavi: ["original","vocals","vocals for transcription",
      "piano","drums","bass","other"]
  """
  # get only filename with and witout ext
  filename: str = os.path.basename(audio_path)  # with ext
  fname, ext = os.path.splitext(filename)

  # folder where spleeter puts vocals and accompaniment tracks
  separated_track_folder = os.path.join(SEPARATED_TRACKS_PATH, fname)

  if not os.path.exists(separated_track_folder):
    # spleeter to separate audio voice from instruments
    separator: Separator = Separator("spleeter:5stems")
    separator.separate_to_file(audio_path, SEPARATED_TRACKS_PATH)

  vocals_for_transcription = os.path.join(separated_track_folder, "vocals_for_transcription.wav")
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


def track_name_to_json_transcription(filepath: str) -> str:
  """
  Prende in input il path di un file che può contenere anche le cartelle
  dove esso è contenuto. Restituisce il nome del file cambiando
  estensione path a quello definito nella costante `JSON_SAVES` e poi
  cambia estensione a .json
  
  Parameters
  ----------
  filepath : str
      file path
  
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
  transcription_json = os.path.join(JSON_SAVES, fname+".json")
  return transcription_json


def transcribe_with_timestamps_whisper(
    file_path: str, 
    print_final_result: bool = False,
):
  """
  Trascrive un file audio in wav con formato mono pcm e 16000 Hz

  Inoltre salva il dizionario delle trascrizioni come json chiamandolo
  come il nome della traccia audio, nella cartella definita dalla
  costante `JSON_SAVES`
  
  Parameters
  ----------

  file_path : str
      Il percorso del file audio WAV da trascrivere.

  only_print_final_result : bool, optional
      Se True, stampa solo il risultato finale della trascrizione (default è False).

  animate : bool, optional
      Se True, mostra la trascrizione in tempo reale durante l'elaborazione (default è False).
  
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
  transcription_json = track_name_to_json_transcription(file_path)
  if os.path.exists(transcription_json):
    with open(transcription_json, "r", encoding="utf-8") as jsonfile:
      transcribed_dict = json.load(jsonfile)
    return transcribed_dict
  

  model = whisper.load_model(
    WHISPER_MODEL, 
    device=DEVICE, 
    download_root=MODELS_PATH
  )
  dict_res = model.transcribe(file_path, word_timestamps=True)

  # remove words with too short duration
  duration_tol: float = 0.20
  dict_res["result"] = [word for segment in dict_res["segments"] for word in segment["words"] if word["end"]-word["start"] > duration_tol]
  dict_res["text"] = " ".join(word_dict["word"] for word_dict in dict_res["result"])

  if print_final_result:
    print(dict_res["text"])

  # save json to JSON_SAVES
  with open(transcription_json, "w", encoding="utf-8") as jsonfile:
    json.dump(dict_res, jsonfile, indent=2)

  return dict_res


def clean_filename(filename: str) -> str:
  """
  Rimuove i seguenti caratteri problematici <span>:"/\\|?* dalla stringa
  `filename` di input, inoltre rimuove anche altri caratteri generalmente
  non permessi nei file (caratteri di controllo ASCII da \\x00 a \\x1F) 
  
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
  filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)
  return filename


def k_best_chunks_with_phonetic_embeddings(
    wav_path: str,
    queries: list[str],
    k: int,
    transcribed_dict: dict | None = None,
# ) -> tuple[list[float],list[float]]:
) -> tuple[np.ndarray, np.ndarray]:
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
  
  # save some time
  if not transcribed_dict:
    transcription_json = track_name_to_json_transcription(wav_path)
    if os.path.exists(transcription_json):
      with open(transcription_json, "r", encoding="utf-8") as jsonfile:
        transcribed_dict = json.load(jsonfile)
    else:
      transcribed_dict = transcribe_with_timestamps_whisper(wav_path)

  whisper_lan: str = transcribed_dict["language"]
  espeak_lan: str = WHISPER_TO_ESPEAK_LAN[whisper_lan]
  # language: str = "en-us"
  print(f"{whisper_lan = }")
  print(f"{espeak_lan = }")

  # Modello per calcolare gli embedding
  model = SentenceTransformer(
    'all-MiniLM-L6-v2',  # FIXME maybe is better an explicit multilingual one
    device=DEVICE, 
    cache_folder=CACHE_SENTENCE_TRANSFORMERS
  )

  transcription = transcribed_dict["text"]
  transcription_words = transcription.split()

  start_times = np.zeros((len(queries),k))
  end_times = np.zeros((len(queries),k))
  transcriptions = np.full((len(queries),k), "", dtype=object)  # str has apparently fixed size

  word_probabilities = np.array([word_dict["probability"] for word_dict in transcribed_dict["result"]])

  for query_idx, query in tqdm(enumerate(queries), desc="Processing Queries", total=len(queries)):
    query = clean_filename(query)
    query_words = query.split()

    len_transcription, len_query = len(transcription_words), len(query_words) 

    contiguous_chunks = [transcription_words[i:i+len_query] for i in range(len_transcription-len_query+1)]

    query_embedding = model.encode(query, convert_to_numpy=True)
    chunks_embeddings = model.encode([" ".join(chunk) for chunk in contiguous_chunks], convert_to_numpy=True)

    similarities = model.similarity(chunks_embeddings, query_embedding).numpy().reshape(-1)
    # cos_sim = util.cos_sim(chunks_embeddings, query_embedding).numpy()

    # calcola similarità fonetica parola-wise
    # # FIXME parte troppo lenta...
    # phonetic_similarities = np.zeros_like(similarities)
    # for i, chunk in tqdm(enumerate(contiguous_chunks), desc="Phonetic similarities", total=len(contiguous_chunks)):
    #   cur_sim = np.zeros(len_query)
    #   for j, (chk_word, query_word) in enumerate(zip(chunk, query_words)):
    #     chk_phonetic = phonemize(chk_word, language="en-us", backend="espeak")
    #     query_phonetic = phonemize(query_word, language="en-us", backend="espeak")
    #     cur_sim[j] = Levenshtein.ratio(chk_phonetic, query_phonetic)
    #   phonetic_similarities[i] = np.mean(cur_sim)
    # # FIXME parte troppo lenta...

    # Speedup 1 ---
    # Supponiamo che query_words e contiguous_chunks siano definiti
    query_phonetics = {word: phonemize(word, language=espeak_lan, backend="espeak") for word in set(query_words)}

    # Pre-calcoliamo le trascrizioni fonetiche di tutti i chunk
    chunk_words = {word for chunk in contiguous_chunks for word in chunk}
    chunk_phonetics = {word: phonemize(word, language=espeak_lan, backend="espeak") for word in chunk_words}

    phonetic_similarities = np.zeros(len(contiguous_chunks))

    for i, chunk in tqdm(enumerate(contiguous_chunks), desc="Phonetic similarities", total=len(contiguous_chunks)):
      cur_sim = np.array([
        Levenshtein.ratio(chunk_phonetics[chk_word], query_phonetics[query_word])
        for chk_word, query_word in zip(chunk, query_words)
      ])
      phonetic_similarities[i] = np.mean(cur_sim)
    # Speedup 1 ---

    # medie delle probabilità delle parole trascritte per ogni chunk
    kernel = np.ones(len_query) / len_query
    chunk_mean_probabilities = np.convolve(word_probabilities, kernel, mode='valid')

    # res_sim = 0.4*similarities + 0.6*phonetic_similarities
    # res_sim = 0.35*similarities + 0.55*phonetic_similarities + 0.1*chunk_mean_probabilities
    res_sim = 0.35*similarities + 0.50*phonetic_similarities + 0.15*chunk_mean_probabilities

    top_k_idx = np.argsort(res_sim)[::-1][:k]
    # best_chunk_idx = np.argmax(res_sim)
    # best_chunk_idx = top_k_idx[0]

    # print(f"Best chunk for query '{query}' is:\n  --> {contiguous_chunks[best_chunk_idx]}")

    # update top k transcriptions for current query
    for i, idx in enumerate(top_k_idx):
      transcriptions[query_idx, i] = " ".join(contiguous_chunks[idx])

    print(f"Top {k} chunks for query '{query}':")
    for i, idx in enumerate(top_k_idx, 1):
      print(f"  {i:>2}. {' | '.join(contiguous_chunks[idx]):<70}\t sim: {res_sim[idx]:.5f}\t cos sim: {similarities[idx]:.5f}\t phonetic sim: {phonetic_similarities[idx]:.5f}\t  prob: {chunk_mean_probabilities[idx]:.5f}")
    print()

    for k_idx, idx in enumerate(top_k_idx):
      chunk_dict = transcribed_dict["result"][idx: idx+len_query]
      start_time: float = chunk_dict[0]["start"]
      end_time: float = chunk_dict[-1]["end"]

      start_times[query_idx, k_idx] = start_time
      end_times[query_idx, k_idx] = end_time

  return start_times, end_times, transcriptions


def full_instruments_ffmpeg_cut(
    start_time: float,
    end_time: float,
    track_names: dict[str, str],
    query: str,
    top_k_position: int,
    selections: dict[str,bool]
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
  query = clean_filename(query)
  track_name: str = os.path.basename(os.path.splitext(track_names["original"])[0])

  # oss from selection is not possible to select original with any other
  keys: list[str] = ["original", "vocals", "piano", "drums", "bass", "other"]
  desc: str = "_".join(k for k in keys if selections[k])
  selected_tracks: list[str] = [track_names[k] for k in keys if selections[k]]

  # ffmpeg export
  track_desc: str = f"{track_name}_{'_'.join(query.split())}_{top_k_position}_{desc}.wav"
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
    
  # os.system(command)
  stream.run()

  # st.write(os.path.basename(out_track_name))
  # st.write(f"time interval : [{start_time},{end_time}]")
  # st.audio(out_track_name)
  # print()

  # doing this only for streamlit ui
  return out_track_name
  

# TODO move in streamlit file
# def streamlit_pipeline(
#     track: str, 
#     # queries: list[str],
#     query: str,
#     k: int = 3,
# ):
#   """
#   TODO for now only a single query, maybe multi query in background
#   """

#   with time_it("Audio Separation..."):
#     with st.spinner("Audio Separation..."):
#       track_names: dict[str,str] = separate_all_tracks(track)

#   with time_it("Transcription..."):
#     with st.spinner("Transcription..."):
#       transcribed_dict = transcribe_with_timestamps_whisper(track_names["vocals for transcription"])

#   with time_it("Chunking..."):
#     with st.spinner("Chunking..."):
#       start_times, end_times, transcriptions = k_best_chunks_with_phonetic_embeddings(
#         track_names["vocals"], 
#         [query], 
#         k,
#         transcribed_dict
#       )

#   with time_it("Final ffmpeg cut"):
#     with st.spinner("Final audio cut..."):
#       top_extracted_cuts: list[str] = [""]*(k)
#       for i in range(k):
#         top_extracted_cuts[i] = full_instruments_ffmpeg_cut(
#           start_times[0,i],
#           end_times[0,i],
#           track_names,
#           query,
#           i+1,
#           selections,
#         )

#   # doing this only for streamlit ui
#   start_end_times = list(zip(start_times[0,:], end_times[0,:]))
#   return track_names, top_extracted_cuts, start_end_times, transcriptions[0,:]
