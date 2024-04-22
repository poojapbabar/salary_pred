from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipeline.predication_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint(): 
    if request.method == 'GET': 
        return render_template('form.html')
    else: 
        age = request.form.get('age')
        educational_num = request.form.get('educational-num')
        hours_per_week = request.form.get('hours_per_week')
        occupation = request.form.get('occupation')
        workclass = request.form.get('workclass')

        if age is not None and educational_num is not None and hours_per_week is not None:
            try:
                data = CustomData(
                    age=int(age),
                    educational_num=int(educational_num),
                    hours_per_week=int(hours_per_week), 
                    occupation=occupation, 
                    workclass=workclass
                )
                new_data = data.get_data_as_dataframe()
                predict_pipeline = PredictPipeline()
                pred = predict_pipeline.predict(new_data)

                results = round(pred[0], 2)

                return render_template('results.html', final_result=results)
            except ValueError:
                return "Please provide valid integer values for age, educational number, and hours per week."
        else:
            return "Please provide values for age, educational number, and hours per week."

if __name__ == '__main__': 
    app.run(host='0.0.0.0', debug=True)
