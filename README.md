# YouTube Commons Project for Moroccan Darija
This project automates the download and transcription of Creative Commons (CC-BY) licensed videos from selected Moroccan YouTube channels. The audio content is transcribed using Gemini with support for code-switching between Darija, French, and English. Proper credit is given to all content creators.

## Features
- **Automated Download**: Fetches CC-BY licensed videos only.
- **Accurate Transcription**: Uses Gemini for multi-language transcription.
- **Content Crediting**: Ensures content creators are credited for their work.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the script to process all listed channels:
```bash
python youtube_cc_by_scraper.py
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
non-sense...
| Channel URL                                         | Description                                          | CC-BY | Scraped |
|----------------------------------------------------|------------------------------------------------------|-------|---------|
| https://www.youtube.com/@SoufianeDanilo         | Sarcasm, Morocco news, funny content               | ✅    | ❌      |
**Legend:**
- ✅: Successfully scraped
- ❌: Not scraped yet
- On going: Script is running

## Other data sources
- https://huggingface.co/datasets/BrunoHays/wikitongues-darija?row=0: high quality, manually annotated but less than 10 minutes

## License
This project is under the MIT License. Only CC-BY licensed videos are downloaded to respect copyright laws.

## Disclaimer
Videos flagged as harmful or inappropriate are excluded from processing.
