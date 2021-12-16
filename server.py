#!/usr/bin/env python
# coding: utf-8

# pip install flask

import numpy as np
from PIL import Image, UnidentifiedImageError
from feature_extracer import FeatureExtracer
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
#get_ipython().run_line_magic('run', 'feature_extracer.ipynb import FeatureExtracter # Adding Jupyter Notebook file as class')
import requests
import random



app = Flask(__name__)

fe = FeatureExtracer()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem +  ".jpg"))
features = np.array(features)

@app.route("/", methods=["GET", "POST"])
def index():
    #return "Server is running!"
    #return render_template("index.html")

    getparam = request.args.get('q') 

    if request.method == "POST":
        
        # Get image as file
        file = request.files["query_img"]
        #save query image
        img = Image.open(file.stream) # PIL image
        upload_img_path = "static/upload/" + datetime.now().isoformat().replace(":", "-").replace(".", "-") + "_" +  file.filename
        img.save(upload_img_path)
        print(img)
        # Run our search 
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1) # L2 distances to the features
        #print(dists)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]
        
        #print(scores)

        # Return data to render and set the scores
        
        return render_template("index.html", query_path=upload_img_path, scores=scores)
    
    elif getparam:

        query = getparam
        getsearch = getparam

        r = requests.get("https://api.qwant.com/v3/search/images",
            params={
                'count': 3,
                'q': query,
                't': 'images',
                'safesearch': 1,
                'locale': 'de_de',
                'offset': 0,
                'device': 'desktop',
                'thumb_type': 'jpg'
            },
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
            }
        )

        response = r.json().get('data').get('result').get('items')
        urls = [r.get('media') for r in response]
        endurl = random.choice(urls)

        # Check for error - if img file is not an RGB file - in case error 500
        try:
            img = Image.open(requests.get(endurl, stream=True).raw)
        except UnidentifiedImageError:
            endurl = random.choice(urls)
            img = Image.open(requests.get(endurl, stream=True).raw) 
        #file = request.files[img]
        #save query image
        #img = Image.open(file.stream) # PIL image
        upload_img_path = "static/upload/" + datetime.now().isoformat().replace(":", "-").replace(".", "-") + "_" +  'image.jpg'
        #img.save(upload_img_path)
        img.save(upload_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1) # L2 distances to the features
        #print(dists)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

       

        return render_template("index.html", query_search=getsearch, scores=scores)
        #return getparam
    else:   
        return render_template("index.html")

if __name__ == "__main__":
     app.run() # debug=True ,port=8080,use_reloader=False

