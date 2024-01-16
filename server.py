from flask import Flask, request, render_template ,url_for
import utils
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    celebrity_probabilities = [0.0, 0.0, 0.0, 0.0]
    return render_template('index.html', celebrity_probabilities = celebrity_probabilities)

@app.route('/classifier', methods=['GET','POST'])
def classifier():
    celebrity_probabilities = [0.0, 0.0, 0.0, 0.0]
    if request.method=='POST':
        if 'image' in request.files:
            image_data = request.files['image']
            prediction = utils.classify(image_data )
            if type(prediction) is str:
                return render_template('index.html', final_result = prediction,  celebrity_probabilities = prediction   )
            else:
                print(type(prediction))
                response = np.round(prediction,2)
                if response[0] == 1:
                    result = "Alexandra Daddario"
                elif response[1] == 1:
                    result = "Johnny Depp"
                elif response[2] == 1:
                    result = "Mahendra Singh Dhoni"
                elif response[3] ==1:
                    result = "Narendra Modi"
                return render_template('index.html', final_result = result,  celebrity_probabilities = prediction   )
        else:
            return "Please upload an image." 
    else:
        return render_template('index.html', celebrity_probabilities = celebrity_probabilities)

if __name__ == "__main__":
    utils.load_artifacts()
    app.run(debug=True)
