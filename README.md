# IngressSwagReconition
IngressSwagReconition is a Python application available through a webpage that identify an Ingress patch, after taking or uploading a picture it.

## How to use
1) Open the index.html page from the local / remote server and scan the patch with your phone camera or upload a picture
2) Image is uploaded to the "scans/unsorted" folder in the server
3) The patch is analysed by the "SwagAnalysis.py" script that returns the best prediction
4) If the user knows the patch clicking on the corresponding button will improve future predictions:
    - Correct -> The patch will be moved in the to the "scans/class" folder
    - Unsure -> The patch will stay in the "scans/unsorted" folder
    - Incorrect ->  The patch will be moved in the to the "unidentified" folder
5) Go to the admin panel and click on "Re-train model" to train the model again with the new addition in the dataset

## How to deploy
0) Create a new virtual python environment (optional)
1) Install dependancies (flask, flask_cors, pil, cv2, numpy)
2) Run runServ.py
3) The website, even if ran locally, must be open through https. Allow exception on your browser or generate self-signed certificates.

## Roadmap
- Write a data policy
- Improve quality of scan (currently all images are resized to 720x720px)
- Split classes in different categories (RES, ENL, Event, etc.)
- Allow admin panel to:
    - Move / delete scans
    - Create a new class
- Create API endopoints