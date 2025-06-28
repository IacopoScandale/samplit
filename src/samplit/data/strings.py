"""
This script contains all the constants of the project, including
strings, paths, hyperparameters and more
"""
import os

from torch.cuda import is_available


MODELS_PATH: str = "pretrained_models"  # spleeter (tensorflow) default path
TRACKS_PATH: str = "tracks"
SEPARATED_TRACKS_PATH: str = os.path.join(TRACKS_PATH, "separated_tracks")
SLICES_FOLDER: str = os.path.join(TRACKS_PATH, "extracted_slices")
JSON_SAVES: str = os.path.join(MODELS_PATH, "json_saved_transcriptions")
CACHE_HUGGING_FACE: str = os.path.join(MODELS_PATH, "hugging_face")
DEVICE: str = "cuda" if is_available() else "cpu"
WHISPER_MODEL: str = "medium"  # tiny, base, small, medium, large, turbo
PHONEMIZE_BACKEND: str = "espeak"
SPLEETER_MODEL_PIPELINE: str = "spleeter:5stems"
PLOTS_FOLDER: str = "plots"
SNS_PALETTE: str = "Set2"
TITLE_FONTSIZE: int = 22
TITLE_PAD: int = 20
TEXT_FONTSIZE: int = 16
SIM_PARAMS: tuple[float] = (0.35, 0.50, 0.15)
ESPEAK_NG_DLL: str = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
"""
windows full path of `libespeak-ng.dll` file.

default: `C:\Program Files\eSpeak NG\libespeak-ng.dll`
"""

# create folders
if not os.path.exists(TRACKS_PATH):
  os.mkdir(TRACKS_PATH)
if not os.path.exists(SEPARATED_TRACKS_PATH):
  os.mkdir(SEPARATED_TRACKS_PATH)
if not os.path.exists(SLICES_FOLDER):
  os.mkdir(SLICES_FOLDER)
if not os.path.exists(MODELS_PATH):
  os.mkdir(MODELS_PATH)
if not os.path.exists(JSON_SAVES):
  os.mkdir(JSON_SAVES)
if not os.path.exists(PLOTS_FOLDER):
  os.mkdir(PLOTS_FOLDER)


DATASETS_FOLDER: str = "datasets"
"""
datasets folder path
"""

# jamendo dataset strings
JAMENDO_DIR: str = os.path.join(DATASETS_FOLDER, "jamendo_dataset")
"""
jamendo dataset folder path
"""
J_LYRICS_CSV: str = os.path.join(JAMENDO_DIR, "JamendoLyrics.csv")
"""
path to the csv file with all the song information
"""
J_MP3_DIR: str = os.path.join(JAMENDO_DIR, "mp3")
"""
path to the folder with all the jamendo mp3 files
"""
J_LINES_DIR: str = os.path.join(JAMENDO_DIR, "annotations", "lines")
"""
path to the folder with all the jamendo queries
"""
J_TEMP_CUTS_DIR: str = os.path.join(JAMENDO_DIR, "temp_cuts")
"""
path to the folder with all the jamendo temporary cuts
"""
J_MODEL_LINES: str = os.path.join(JAMENDO_DIR, "model_lines")
"""
path to the folder with all the custom lines files with model answers
"""
J_LYRICS_DIR: str = os.path.join(JAMENDO_DIR, "lyrics")
"""
path to the folder with all the jamendo lyrics
"""
J_WORDS_DIR: str = os.path.join(JAMENDO_DIR, "annotations", "words")
"""
path to the folder with all the jamendo words start and end times
"""
J_CUSTOM_LINES_DIR: str = os.path.join(JAMENDO_DIR, "annotations", "custom_lines")
"""
path to the folder with all the custom lines
"""
J_DATAFRAME_CSV: str = os.path.join(JAMENDO_DIR, "jamendo_final_df.csv")
"""
path to the jamendo dataframe with all info and model answers, ready to
make plots
"""
J_EVALUATION_DIR: str = os.path.join(JAMENDO_DIR, "my_evaluation_df")
"""
path to the folder containing evaluation csv files
"""
# create jamendo custom folders only if dataset folder is present
if os.path.exists(J_LINES_DIR):
  if not os.path.exists(J_TEMP_CUTS_DIR):
    os.mkdir(J_TEMP_CUTS_DIR)
  if not os.path.exists(J_LINES_DIR):
    os.mkdir(J_LINES_DIR)
  if not os.path.exists(J_CUSTOM_LINES_DIR):
    os.mkdir(J_CUSTOM_LINES_DIR)
  if not os.path.exists(J_MODEL_LINES):
    os.mkdir(J_MODEL_LINES)
  if not os.path.exists(J_EVALUATION_DIR):
    os.mkdir(J_EVALUATION_DIR)


MAX_CHAR_NUM: int = 1_000_000


# Whisper fine-tuning
WHISPER_PRETRAINED: str = "openai/whisper-medium"
SPLEETER_MODEL_PREPROCESS: str = "spleeter:2stems"
"""
Use this model only to separate voice
"""
J_SEPARATED_DIR: str = os.path.join(JAMENDO_DIR, "voice_only")
if (os.path.exists(JAMENDO_DIR)) and (not os.path.exists(J_SEPARATED_DIR)):
  os.mkdir(J_SEPARATED_DIR)
TEST_SIZE: float = 0.2
SEED: int = 42
WHISPER_FINETUNED_MODEL_DIR: str = os.path.join(MODELS_PATH, "whisper_finetuned_model")
WHISPER_FINETUNED_PROCESSOR_DIR: str = os.path.join(
  MODELS_PATH, "whisper_finetuned_processor"
)
LOGGING_DIR: str = os.path.join(MODELS_PATH, "logs")


WHISPER_TO_ESPEAK_LAN: dict[str, str] = {
  "en": "en-us",
  "de": "de",
  "es": "es",
  "zh": "cmn",  # Chinese (Mandarin)
  "ru": "ru",
  "ko": "ko",
  "fr": "fr-fr",
  "ja": "ja",
  "pt": "pt",
  "tr": "tr",
  "pl": "pl",
  "ca": "ca",
  "nl": "nl",
  "ar": "ar",
  "sv": "sv",
  "it": "it",
  "id": "id",
  "hi": "hi",
  "fi": "fi",
  "vi": "vi",  # Vietnamese (Northern)
  "he": "he",
  "uk": "uk",
  "el": "el",
  "ms": "ms",
  "cs": "cs",
  "ro": "ro",
  "da": "da",
  "hu": "hu",
  "ta": "ta",
  "no": "nb",  # Norwegian Bokmål
  "th": "th",
  "ur": "ur",
  "hr": "hr",
  "bg": "bg",
  "lt": "lt",
  "la": "la",
  "mi": "mi",
  "ml": "ml",
  "cy": "cy",
  "sk": "sk",
  "te": "te",
  "fa": "fa",
  "lv": "lv",
  "bn": "bn",
  "sr": "sr",
  "az": "az",
  "sl": "sl",
  "kn": "kn",
  "et": "et",
  "mk": "mk",
  "br": "cy",  # breton to welsh
  "eu": "eu",
  "is": "is",
  "hy": "hy",
  "ne": "ne",
  "mn": "my",  # Myanmar (Burmese)
  "bs": "bs",
  "kk": "kk",
  "sq": "sq",
  "sw": "sw",
  "gl": "pt",  # galician to portuguese
  "mr": "mr",
  "pa": "pa",
  "si": "si",
  "km": "th",  # khmer to thai
  "sn": "sw",  # shona to swahili
  "yo": "sw",  # yoruba to swahili
  "so": "ar",  # somali to arabic
  "af": "af",
  "oc": "ca",  # occitan to catalan
  "ka": "ka",
  "be": "be",
  "tg": "fa",  # tajik to persian
  "sd": "sd",
  "gu": "gu",
  "am": "am",
  "yi": "de",  # yiddish to german
  "lo": "th",  # lao to thai
  "uz": "uz",
  "fo": "is",  # faroese to icelandic
  "ht": "ht",
  "ps": "fa",  # pashto to persian
  "tk": "tk",
  "nn": "nb",  # Nynorsk to Norwegian Bokmål
  "mt": "mt",
  "sa": "hi",  # sanskrit to hindi
  "lb": "lb",
  "my": "my",
  "bo": "cmn",  # tibetan to chinese mandarin
  "tl": "id",  # tagalog to indonesian
  "mg": "id",  # malagasy to indonesian
  "as": "as",
  "tt": "tt",
  "haw": "haw",
  "ln": "sw",  # lingala to swahili
  "ha": "ar",  # hausa to arabic
  "ba": "ba",
  "jw": "id",  # javanese to indonesian
  "su": "id",  # sudanese to indonesian
  "yue": "yue",
}
"""
Semi-automatically aligned language mapping from the Whisper detected 
language to the corresponding eSpeak-NG language.
"""