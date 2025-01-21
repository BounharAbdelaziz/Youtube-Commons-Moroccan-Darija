# model to use for transcription
MODEL_NAME = "gemini-2.0-flash-exp"
# MODEL_NAME = "gemini-exp-1206"

idx = 1 if MODEL_NAME == "gemini-2.0-flash-exp" else 2
TOPIC='PODCAST_LOT_CS'

# Path to dataset where we save
model_used = "gemini-2-0-flash-exp" if MODEL_NAME == "gemini-2.0-flash-exp" else MODEL_NAME
HF_DATA_PATH = f"BounharAbdelaziz/YCM-{model_used}-{TOPIC}"
print(f'[INFO] Data will be stored in: {HF_DATA_PATH}')

# Access token to the dataset
TOKEN = "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"
# first time we create dataset
IS_FIRST_TIME = True
# IS_FIRST_TIME = False
# maximum number of videos per channel
MAX_VIDEOS = 2_000

MAX_WAIT_TIME_SECONDS = 3600 * 1 # 1 hour maximum


# maximum number of retrial of resource exhausted
MAX_RETRIES_GEMINI_TRANSCRIBE = 60 * 60 * 24 * 3 # assuming 1 retry per second, we wait until the next 3rd day. SHould be more than enough as things refresh per day

# path to temporary downloaded audio files
DOWNLOAD_PATH = f"./downloaded_audios_{idx}_{TOPIC}"

# # List of YouTube channel URLs to process
# channel_urls = [
#     # "https://www.youtube.com/@-ayaat5191",                          # some of Yassine El Amri talks in cc-by
#     # "https://www.youtube.com/@Amiremmal",                           # finance, trading in darija. contains some code-switching
#     # "https://www.youtube.com/@achrafeljraoui",                      # one podcast about studies in france. contains some code-switching
#     # "https://www.youtube.com/@wladlhajexperience",                  # podcast on general stuff in darija. contains some code-switching
#     # "https://www.youtube.com/@SGHACAST",                            # podcast with a lot of code-switching
#     "https://www.youtube.com/@storiesbdarijareal",                  # paranormal stories in darija
#     "https://www.youtube.com/@BIMO1",                               # general topics, few videos cc-by
#     # "https://www.youtube.com/@Le360.",                              # news, stories, etc
#     # "https://www.youtube.com/@mramiin3274",                         # trading, e-commerce iun darija
#     # "https://www.youtube.com/@youssefassim",                        # mostly rap news
#     # "https://www.youtube.com/@StreetartTV",                         # rap news
#     # "https://www.youtube.com/@SoufianeDanilo",                      # sacrasm, morocco news, funny stuff, etc
#     # "https://www.youtube.com/@branimoxgamer404",                    # rap news and other news.
#     # "https://www.youtube.com/@elamri.page.officielle",            # islamic talks. non cc
#     # "https://www.youtube.com/@Dr_Lahlou",                         # islamic talks. non cc
#     # "https://www.youtube.com/@janarbatia",                        # harmfull content. non cc
#     # "https://www.youtube.com/@naoufalchaara",                     # history, culture in darija. non cc
#     # "https://www.youtube.com/@AzizBadir",                         # stories, culture in darija. non cc
#     # "https://www.youtube.com/@Mustaphaswingaofficiel",            # general topics in darija. non cc
#     # "https://www.youtube.com/@Ghir7na",                           # news, sacrasm. non cc
#     "https://www.youtube.com/@Choumicha",                           # the one and only choumicha. Moroccan cuisine in darija. non cc
#     "https://www.youtube.com/@Kick.Bdarija",                        # harmfull content
#     # "https://www.youtube.com/@canal_opium_school",                  # studies in darija
# ]

# List of YouTube channel URLs to process
channel_urls = [
    # "https://www.youtube.com/@-ayaat5191",                          # some of Yassine El Amri talks in cc-by
    # "https://www.youtube.com/@Amiremmal",                           # finance, trading in darija. contains some code-switching
    # "https://www.youtube.com/@achrafeljraoui",                      # one podcast about studies in france. contains some code-switching
    # "https://www.youtube.com/@wladlhajexperience",                  # podcast on general stuff in darija. contains some code-switching
    "https://www.youtube.com/@SGHACAST",                            # podcast with a lot of code-switching
    # "https://www.youtube.com/@storiesbdarijareal",                  # paranormal stories in darija
    # "https://www.youtube.com/@BIMO1",                               # general topics, few videos cc-by
    # "https://www.youtube.com/@Le360.",                              # news, stories, etc
    # "https://www.youtube.com/@mramiin3274",                         # trading, e-commerce iun darija
    # "https://www.youtube.com/@youssefassim",                        # mostly rap news
    # "https://www.youtube.com/@StreetartTV",                         # rap news
    # "https://www.youtube.com/@SoufianeDanilo",                      # sacrasm, morocco news, funny stuff, etc
    # "https://www.youtube.com/@branimoxgamer404",                    # rap news and other news.
    # "https://www.youtube.com/@elamri.page.officielle",            # islamic talks. non cc
    # "https://www.youtube.com/@Dr_Lahlou",                         # islamic talks. non cc
    # "https://www.youtube.com/@janarbatia",                        # harmfull content. non cc
    # "https://www.youtube.com/@naoufalchaara",                     # history, culture in darija. non cc
    # "https://www.youtube.com/@AzizBadir",                         # stories, culture in darija. non cc
    # "https://www.youtube.com/@Mustaphaswingaofficiel",            # general topics in darija. non cc
    # "https://www.youtube.com/@Ghir7na",                           # news, sacrasm. non cc
    # "https://www.youtube.com/@Choumicha",                           # the one and only choumicha. Moroccan cuisine in darija. non cc
    # "https://www.youtube.com/@Kick.Bdarija",                        # harmfull content
    # "https://www.youtube.com/@canal_opium_school",                  # studies in darija
]

# Folders
downloads_folder = f"downloads_{idx}_{TOPIC}"
chunks_folder = f"chunks_{idx}_{TOPIC}"
transcripts_folder = f"transcripts_{idx}_{TOPIC}"

SYSTEM_INSTRUCTION = """You are an expert transcriber specializing in Moroccan Darija (الدارجة المغربية) with a meticulous eye for code-switching. 
                        Your sole task is to provide verbatim transcriptions of spoken Moroccan Darija from audio files. 
                        The audios might contain code-switching, primarily with French and English. 
                        When code-switching occurs, you MUST transcribe the words in their original script (e.g., Latin script for French/English). 
                        Note that you do not need to use capital letters if the code-switching occurs in the middle of a sentence. 
                        Immediately after the code-switched portion, revert to using Arabic script (الحروف العربية) for the Moroccan Darija.
                        Crucially, you must accurately capture and represent all instances of code-switching with French and English words.
                        Do not translate or explain anything. Provide only the verbatim transcription, accurately reflecting the input audio and language switching.

                        Examples:

                        - If the audio says: "غادي نمشي نشوف un film", the transcription should be: "غادي نمشي نشوف un film"
                        - If the audio says: "I have a meeting, من بعد نهضرو", the transcription should be: "I have a meeting, من بعد نهضرو"
                        - If the audio says: "صافي, نتسناك", the transcription should be: "صافي, نتسناك"
                        - If the audio says: "صافي, d'accord, غادي نتسناك", the transcription should be: "صافي, d'accord, غادي نتسناك"
                    """