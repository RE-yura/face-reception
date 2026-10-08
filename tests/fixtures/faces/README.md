# Face fixtures

Official NASA astronaut portraits from Wikimedia Commons. As works of NASA they are in the public domain.
`scripts/fetch-fixtures.mjs` downloads the 960px versions and keeps the top 960×720 (the face is in the upper part).

| File | Person | Source |
|---|---|---|
| `meir-a.jpg` | Jessica Meir (2025) | https://commons.wikimedia.org/wiki/File:Official_portrait_of_NASA_astronaut_Jessica_Meir_wearing_a_spacesuit_(jsc2025e078605_alt).jpg |
| `meir-b.jpg` | Jessica Meir (2018, black and white) | https://commons.wikimedia.org/wiki/File:Jessica_Meir_official_portrait_in_an_EMU_(B%26W).jpg |
| `kim.jpg` | Jonny Kim (2024) | https://commons.wikimedia.org/wiki/File:Jsc2024e052605_alt_(Aug._6,_2024)_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg |

`../opencv-reference.json` holds OpenCV's own YuNet + SFace results for these files (`scripts/opencv_reference.py`).
