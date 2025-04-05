import os
import torch

# folders
MODELS_PATH: str = "pretrained_models"  # uso questo solo perché è il default di spleeter con tensorflow
TRACKS_PATH: str = "tracks"
SEPARATED_TRACKS_PATH: str = os.path.join(TRACKS_PATH, "separated_tracks")
SLICES_FOLDER: str = os.path.join(TRACKS_PATH, "extracted_slices")
JSON_SAVES: str = os.path.join(MODELS_PATH, "json_saved_transcriptions")
CACHE_SENTENCE_TRANSFORMERS: str = os.path.join(MODELS_PATH, "hugging_face")

# TODO verificare se ci siano da aggiungere altre linee di codice per 
# spostare altro sul device (non l'ho provato :)
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL: str = "medium"  # scegli tra tiny, base, small, medium, large



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
  "yue": "yue"
}