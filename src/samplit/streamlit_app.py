"""
This script contains the streamlit web app. Use the following command to
run: `streamlit run src/samplit/streamlit_app.py`
"""
import os
import zipfile

import streamlit as st
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from data.strings import (
  DEVICE,
  TRACKS_PATH,
  ESPEAK_NG_DLL,
)
from data.utils import time_it
from pipeline_functions import (
  full_instruments_ffmpeg_cut,
  k_best_chunks_with_phonetic_embeddings,
  separate_all_tracks,
  transcribe_with_timestamps_whisper,
)

# espeak-ng windows: https://bootphon.github.io/phonemizer/install.html
if os.name == "nt":
  EspeakWrapper.set_library(ESPEAK_NG_DLL)


print(f"{DEVICE = }")


def main() -> None:
  def streamlit_pipeline(
    track: str,
    # queries: list[str],
    query: str,
    k: int = 3,
  ):
    with time_it("Audio Separation..."):
      with st.spinner("Audio Separation..."):
        track_names: dict[str, str] = separate_all_tracks(track)

    with time_it("Transcription..."):
      with st.spinner("Transcription..."):
        transcribed_dict = transcribe_with_timestamps_whisper(
          track_names["vocals for transcription"]
        )

    with time_it("Chunking..."):
      with st.spinner("Chunking..."):
        start_times, end_times, transcriptions = k_best_chunks_with_phonetic_embeddings(
          track_names["vocals"], [query], k, transcribed_dict
        )

    with time_it("Final ffmpeg cut"):
      with st.spinner("Final audio cut..."):
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

    # following code only for streamlit ui
    start_end_times = list(zip(start_times[0, :], end_times[0, :]))
    return track_names, top_extracted_cuts, start_end_times, transcriptions[0, :]

  # Streamlit GUI ------------------------------------------------------
  def st_blank_line():
    st.markdown("<br>", unsafe_allow_html=True)

  def format_time(seconds: float) -> tuple[str, ...]:
    """
    Examples
    --------
    >>> format_time(120.54)
        ("2:00","54")
    """
    minutes, seconds = divmod(seconds, 60)
    res = f"{int(minutes)}:{seconds:05.2f}"  # Ensures two decimal places
    return tuple(res.split("."))

  st.title("Samplit")
  st.subheader("Phonetic-Based Audio Sampling via Source Separation and Transcription")
  # st.title("Phonetic-Based Audio Sampling via Source Separation and Transcription")
  # st.markdown("### **How It Works:**")
  # with st.expander("How It Works"):
  st.markdown(
    """
    1. **Upload an Audio File:** Choose the audio track to use  
    2. **Enter a Search Query:** Type the specific phrase you want to 
    extract from the audio.  
    3. **Set the Top-k Best Cuts:** Specify how many extractions (best 
    cuts) the model will display.
    4. **Select Instruments (Optional):** Pick any specific instruments 
    you'd like to isolate or focus on.  
    5. **Export Results (Optional):** Download the processed audio or 
    extracted data if needed.
    """
  )

  st_blank_line()
  with st.container(border=True):
    st.subheader("1. Upload an Audio File:")
    uploaded_file = st.file_uploader("Upload an audio file here")

    # avoid if block and stop execution
    if not uploaded_file:
      st.stop()

    # if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    audio_path: str = os.path.join(TRACKS_PATH, uploaded_file.name)
    # write the file locally
    if not os.path.exists(audio_path):
      with open(audio_path, "wb") as file:
        file.write(uploaded_file.read())

    st.audio(audio_path)

  st_blank_line()
  with st.container(border=True):
    st.subheader("2. Enter words to extract:")
    query = st.text_input(
      "Type the query you wish to find in the audio file:",
      help="You cannot leave blank this field",
      placeholder="Enter some words to continue",
    )

    # avoid if block and stop execution
    if not query.strip():
      st.stop()

  st_blank_line()
  with st.container(border=True):
    st.subheader("3. Set the Top-k Best Cuts")

    # Provide a clear explanation for the user
    st.markdown("""
    Enter a value to specify how many top query extractions the model will
    display.  
    For example, if 3 is chosen, the model will show the top 3 most 
    relevant results.
    """)

    # Input field for the integer k with a helpful tooltip
    k = st.number_input(
      "Enter the value:",
      min_value=1,
      step=1,
      value=None,
      format="%d",
      help="Number of top query results to display.",
    )
    # avoid if block and stop execution
    if not k:
      st.stop()

  st_blank_line()
  with st.container(border=True):
    # Create radio buttons
    ORIGINAL_TRACK: str = "Original Track (best quality)"
    PRECISE_SELECTION: str = "Precise Selection:"

    st.subheader("4. Select Instruments:")
    option = st.radio(
      "Choose what to export:",
      [ORIGINAL_TRACK, PRECISE_SELECTION],
      # horizontal=True,
    )

    # Indent checkboxes using columns
    col1, col2 = st.columns([0.05, 0.95])  # Indent checkboxes using columns

    with col2:  # Place checkboxes inside indented column
      choice1 = st.checkbox("Vocals", disabled=(option != PRECISE_SELECTION))
      choice2 = st.checkbox("Piano", disabled=(option != PRECISE_SELECTION))
      choice3 = st.checkbox("Drums", disabled=(option != PRECISE_SELECTION))
      choice4 = st.checkbox("Bass", disabled=(option != PRECISE_SELECTION))
      choice5 = st.checkbox("Other", disabled=(option != PRECISE_SELECTION))

    # selection_keys = ["original", "vocals", "piano", "drums", "bass", "other"]
    selections: dict[str, bool] = {
      "original": option == ORIGINAL_TRACK,
      "vocals": choice1 and option == PRECISE_SELECTION,
      "piano": choice2 and option == PRECISE_SELECTION,
      "drums": choice3 and option == PRECISE_SELECTION,
      "bass": choice4 and option == PRECISE_SELECTION,
      "other": choice5 and option == PRECISE_SELECTION,
    }

  # manually restart
  # st.session_state.button_pressed = False
  # st.stop()

  def reset():
    st.session_state.button_pressed = False
    # clear session state
    if "top_extracted_cuts" in st.session_state:
      # remove cuts from disk
      for track in st.session_state.top_extracted_cuts:
        os.remove(track)
      st.session_state.pop("top_extracted_cuts", None)

    if "start_end_times" in st.session_state:
      st.session_state.pop("start_end_times", None)
    if "transcriptions" in st.session_state:
      st.session_state.pop("transcriptions", None)
    if "checkboxes" in st.session_state:
      st.session_state.pop("checkboxes", None)
    if "audios" in st.session_state:
      st.session_state.pop("audios", None)
    if "btn_label" in st.session_state:
      st.session_state.pop("btn_label")
    if "running" in st.session_state:
      st.session_state.pop("running", None)
    st.rerun()

  st_blank_line()
  # centro l'output in 80% dello spazio: lo distingue dall'input secondo me
  _, col, _ = st.columns([0.1, 0.8, 0.1])

  with col:
    # stato che ricorda se abbiamo già separato le tracce per evitare
    # problemi per come è costruito streamlit che fa rerun quando si
    # modificano checkbox ecc
    if "button_pressed" not in st.session_state:
      st.session_state.button_pressed = False

    if "btn_label" not in st.session_state:
      st.session_state.btn_label = "Separate Words From Audio!"
    separate_btn = st.button(st.session_state.btn_label, use_container_width=True)

    # stop when clicking again
    if "running" in st.session_state:
      if st.session_state.running:
        reset()

    # avoid if block and stop execution
    if not separate_btn and not st.session_state.button_pressed:
      st.stop()

    st.session_state.btn_label = "Reset"
    if separate_btn and st.session_state.button_pressed:
      # print("reset from here")
      reset()

    # non eseguire la pipeline quando si aggiornano i widget e streamlit fa rerun
    if not st.session_state.button_pressed:
      st.session_state.running = True
      (
        st.session_state.track_names,
        st.session_state.top_extracted_cuts,
        st.session_state.start_end_times,
        st.session_state.transcriptions,
      ) = streamlit_pipeline(audio_path, query, k)
      st.session_state.running = False
      st.session_state.checkboxes = [False] * k
      st.session_state.button_pressed = True
      st.session_state.rerun_once = True

    for track_idx in range(k):
      transcription = st.session_state.transcriptions[track_idx]
      # case where k is bigger than the available extracted tracks
      if not transcription:
        break
      track_path = st.session_state.top_extracted_cuts[track_idx]
      start_time, end_time = st.session_state.start_end_times[track_idx]

      track_name = os.path.basename(track_path)
      with st.container(border=True):
        f_start_time, f_start_ms = format_time(start_time)
        f_end_time, f_end_ms = format_time(end_time)
        st.subheader(f"Cut number {track_idx + 1}")
        # st.write(f' - track: **"{track_name}"**')
        # st.markdown(
        #   f"""
        #   <span style="opacity: 0.5;"> • track: </span>
        #   <span style="opacity: 0.5;"> "{track_name}" </span>
        #   """,
        #   unsafe_allow_html=True
        # )
        st.markdown(
          f"""
          <div style="display: flex; align-items: flex-start;">
            <span style="opacity: 0.5; white-space: nowrap;">• track:</span>
            <span style="margin-left: 0.5em; display: inline-block; 
                        max-width: 80%; text-align: left; opacity: 0.5;
                        word-break: break-word; overflow-wrap: break-word;">
              "{track_name}"
            </span>
          </div>
          """,
          unsafe_allow_html=True,
        )
        st_blank_line()
        # st.write(f' - transcription: **"{transcription}"**')
        st.markdown(
          f"""
          <div style="display: flex; align-items: flex-start;">
            <span style="opacity: 0.5; white-space: nowrap;">• transcription:</span>
            <span style="margin-left: 0.5em; display: inline-block; max-width: 80%; text-align: left;">
              "{transcription}"
            </span>
          </div>
          """,
          unsafe_allow_html=True,
        )
        st_blank_line()
        # st.write(f" - time interval: [{start_time}, {end_time}]")
        st.markdown(
          f"""
          <span style="opacity: 0.5;"> • time interval: </span>
          <span>[ {f_start_time}</span>
          <span style="font-size: 14px; opacity: 0.5;">.{f_start_ms}</span>
          <span>, {f_end_time}</span>
          <span style="font-size: 14px; opacity: 0.5;">.{f_end_ms}</span>
          <span> ]</span>
          """,
          unsafe_allow_html=True,
        )

        # checkbox, audio, download button:
        col1, col2, col3 = st.columns([0.05, 0.70, 0.25])
        with col1:
          st.session_state.checkboxes[track_idx] = st.checkbox(
            "a",  # bug: non scrivere tanto, non lasciare vuoto (warning)
            key=f"chk{track_idx}",
            label_visibility="hidden",
          )

        with col2:
          st.audio(track_path)

        with col3:
          with open(track_path, "rb") as file:
            st.download_button(
              "Download",
              data=file,
              file_name=track_name,
              key=f"dwl{track_idx}",
              use_container_width=True,
            )

        # edit audio expander
        with st.expander("Edit audio cut"):
          col1, col2, col3, col4 = st.columns([0.05, 0.45, 0.45, 0.05])
          with col2:
            start_offset = st.number_input(
              "Start offset",
              min_value=-10.0,
              max_value=10.0,
              step=0.25,
              value=0.0,
              key=f"start_offset_{track_idx}",
              help=(
                "add or remove seconds at the beginning of the cut:\n"
                "e.g. -1 let the cut start one second before"
              ),
            )

          with col3:
            end_offset = st.number_input(
              "End offset",
              min_value=-10.0,
              max_value=10.0,
              step=0.25,
              value=0.0,
              key=f"end_offset_{track_idx}",
              help=(
                "add or remove seconds at the end of the cut:\n"
                "e.g. +1 let the cut end one second after"
              ),
            )

          with st.container():
            col1, col2, col3 = st.columns([0.05, 0.90, 0.05])

            with col2:
              if st.button(
                "Apply Changes", key=f"refresh{track_idx}", use_container_width=True
              ):
                # replace audio with last pipeline function
                new_start_time = start_time + start_offset
                new_end_time = end_time + end_offset
                st.session_state.top_extracted_cuts[track_idx] = (
                  full_instruments_ffmpeg_cut(
                    new_start_time,
                    new_end_time,
                    st.session_state.track_names,
                    query,
                    track_idx + 1,
                    selections,
                  )
                )
                # edit new start and end times
                st.session_state.start_end_times[track_idx] = (
                  new_start_time,
                  new_end_time,
                )

                # FIXME restore offset inputs to 0
                # does not work but it is ok also like this

                # rerun otherwise changes do not apply
                st.rerun()

    if st.button("Download Selected Tracks", use_container_width=True):
      to_download = [
        st.session_state.top_extracted_cuts[idx]
        for idx in range(k)
        if st.session_state.checkboxes[idx]
      ]
      if not to_download:
        st.warning("First select some tracks")
      else:
        # just a string "'query'_cuts_'trackname'.zip" all separated by "_"
        zip_filename: str = (
          f"{'_'.join(query.split())}_cuts_"
          f"{'_'.join(os.path.basename(os.path.splitext(audio_path)[0])).split()}"
          ".zip"
        )
        with zipfile.ZipFile(zip_filename, "w") as zf:
          for filepath in to_download:
            filename: str = os.path.basename(filepath)
            zf.write(filepath, arcname=filename)

        with open(zip_filename, "rb") as zf:
          st.download_button(
            "Download Zip",
            data=zf,
            file_name=zip_filename,
            key=zip_filename,
            use_container_width=True,
          )

        os.remove(zip_filename)

    # to refresh separate button label
    if st.session_state.rerun_once:
      st.session_state.rerun_once = False
      st.rerun()


if __name__ == "__main__":
  main()
