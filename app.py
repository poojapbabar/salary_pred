from flask import Flask, request, render_template
#from src.pipeline.predication_pipeline import CustomData, PredictPipeline
from src.pipeline.predication_pipeline import CustomData,PredictPipeline

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
        # Retrieve form data
        age_str = request.form.get('age')  # Retrieve as string
        
        # Check if age is provided
        if age_str:
            # Convert age to integer
            age = int(age_str)
        else:
            # Handle case when age is not provided
            return "Please provide a value for age."

        # Repeat the same process for other form fields
        educational_num_str = request.form.get('educational-num')
        if educational_num_str:
            educational_num = int(educational_num_str)
        else:
            return "Please provide a value for educational number."

        hours_per_week_str = request.form.get('hours_per_week')
        if hours_per_week_str:
            hours_per_week = int(hours_per_week_str)
        else:
            return "Please provide a value for hours per week."

        occupation = request.form.get('occupation')
        workclass = request.form.get('workclass')

        # Create CustomData object
        data = CustomData(
            age=age,
            educational_num=educational_num,
            hours_per_week=hours_per_week,
            occupation=occupation,
            workclass=workclass
        )
        
        # Get data as DataFrame
        new_data = data.get_data_as_dataframe()
        
        # Create PredictPipeline object
        predict_pipeline = PredictPipeline()
        
        # Perform prediction
        pred = predict_pipeline.predict(new_data)

        # Render template with prediction results
        return render_template('results.html', final_result=pred)


if __name__ == '__main__': 
    app.run(host='0.0.0.0', debug=True)
