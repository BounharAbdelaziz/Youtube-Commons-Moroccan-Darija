# YouTube Commons Project for Moroccan Darija

This project automates the download and transcription of Creative Commons (CC-BY) licensed videos from selected Moroccan YouTube channels. The audio content is transcribed using Gemini with support for code-switching between Darija, French, and English. Proper credit is given to all content creators.

Additionally, this repository includes code for training and evaluating speech-to-text (STT) models for Moroccan Darija, along with a Hugging Face Space for testing them interactively.

## Features

- **Automated Download**: Fetches CC-BY licensed videos only.
- **Accurate Transcription**: Uses Gemini for multi-language transcription.
- **Content Crediting**: Ensures content creators are credited for their work.
- **STT Model Training & Evaluation**: Fine-tune and test STT models on Moroccan Darija.
- **Hugging Face Space**: Provides an interactive interface to test the STT models.

## Speech-to-Text (STT) Models

This repository also includes code for fine-tuning and evaluating speech-to-text models on Moroccan Darija. The models are trained using [Whisper](https://openai.com/research/whisper) and other transformer-based architectures. The goal is to improve transcription accuracy for Moroccan Darija, including code-switching scenarios.

## Test the STT Models

A [Hugging Face Space](https://huggingface.co/spaces/atlasia/Moroccan-Fast-Speech-to-Text-Transcription) is available for testing the trained STT models interactively. You can upload your own audio clips and evaluate transcription quality in real-time!

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Data Collection
Run the script to process all listed channels:
```bash
python data_collection/youtube_cc_by_scraper.py
```

### STT Model Training Collection
Run the script to process all listed channels:
```bash
python train_model/main.py
```

## YouTube Channels List

| Channel URL                                         | Description                                          | CC-BY | Scraped |
|----------------------------------------------------|------------------------------------------------------|-------|---------|
| https://www.youtube.com/@-ayaat5191                | Yassine El Amri talks in CC-BY                       | ✅    |  ✅ |
| https://www.youtube.com/@Amiremmal                | Finance, trading in Darija with code-switching      | ✅    | On going      |
| https://www.youtube.com/@SGHACAST               | Podcast with code-switching                         | ✅    | On going      |
| https://www.youtube.com/@mramiin3274           | Trading, e-commerce in Darija                      | ✅    | On going      |
| https://www.youtube.com/@storiesbdarijareal    | Paranormal stories in Darija                        | ✅    | ✅      |
| https://www.youtube.com/@achrafeljraoui        | Podcast about studies in France with code-switching | ✅    | ❌      |
| https://www.youtube.com/@BIMO1                 | General topics, few videos CC-BY                    | ✅    | ✅     |
| https://www.youtube.com/@canal_opium_school     | Studies in Darija                                   | ✅    | ❌      |
| https://www.youtube.com/@wladlhajexperience     | Podcast on general topics with code-switching      | ✅    | ❌      |
| https://www.youtube.com/@Kick.Bdarija           | Harmful content                                     | ✅    | ❌      |
| https://www.youtube.com/@branimoxgamer404      | Rap news and other news                             | ✅    | ❌      |
| https://www.youtube.com/@youssefassim          | Mostly rap news                                     | ✅    | ❌      |
| https://www.youtube.com/@Le360.                | News, stories                                       | ✅    | ❌      |
| https://www.youtube.com/@StreetartTV           | Rap news                                            | ✅    | ❌      |
| https://www.youtube.com/@elamri.page.officielle | Islamic talks (Non-CC)                              | ❌    | ❌      |
| https://www.youtube.com/@Dr_Lahlou             | Islamic talks (Non-CC)                              | ❌    | ❌      |
| https://www.youtube.com/@janarbatia            | Harmful content (Non-CC)                            | ❌    | ❌      |
| https://www.youtube.com/@naoufalchaara         | History, culture in Darija (Non-CC)                 | ❌    | ❌      |
| https://www.youtube.com/@AzizBadir             | Stories, culture in Darija (Non-CC)                 | ❌    | ❌      |
| https://www.youtube.com/@Mustaphaswingaofficiel | General topics in Darija (Non-CC)                  | ❌    | ❌      |
| https://www.youtube.com/@Ghir7na               | News, sarcasm (Non-CC)                              | ❌    | ❌      |
| https://www.youtube.com/@Choumicha               | Moroccan cuisine in Darija                          |  ❌    | ❌      |

## Bad channels

Non-sense...

| Channel URL                                         | Description                                          | CC-BY | Scraped |
|----------------------------------------------------|------------------------------------------------------|-------|---------|
| https://www.youtube.com/@SoufianeDanilo         | Sarcasm, Morocco news, funny content               | ✅    | ❌      |

**Legend:**
- ✅: Successfully scraped
- ❌: Not scraped yet
- On going: Script is running

## Other Data Sources

- [Wikitongues Darija Dataset](https://huggingface.co/datasets/BrunoHays/wikitongues-darija?row=0): High-quality, manually annotated dataset, but less than 10 minutes.

## Disclaimer

Videos flagged as harmful or inappropriate are excluded from processing.
