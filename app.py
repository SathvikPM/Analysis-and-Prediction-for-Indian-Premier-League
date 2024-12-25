import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('model/ipl_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/win', methods=["GET", "POST"])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == 'POST':
        city = request.form['city']
        Home = request.form['Home']
        Away = request.form['Away']
        toss_winner = request.form['toss_winner']
        toss_decision = request.form['toss_decision']
        venue = request.form['venue']

        if toss_winner == 'Home Team':
            toss_winner = Home
        else:
            toss_winner = Away

        input_variables = pd.DataFrame([[city, Home, Away, toss_winner, toss_decision, venue]], columns=['city', 'Home', 'Away', 'toss_winner',
        'toss_decision', 'venue'], dtype=object)

        input_variables.Home.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                      np.arange(0, 14), inplace=True)
        input_variables.Away.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                      np.arange(0, 14), inplace=True)
        input_variables.toss_winner.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                             'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                             'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                             'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                             'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                              np.arange(0, 14), inplace=True)
        input_variables.toss_decision.replace(['bat', 'field'], [0, 1], inplace=True)
        input_variables.city.replace(['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
        'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
        'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
        'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
        'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
        'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah'],
        np.arange(0, 30), inplace=True)
        input_variables.venue.replace(['Rajiv Gandhi International Stadium, Uppal',
        'Maharashtra Cricket Association Stadium',
        'Saurashtra Cricket Association Stadium', 'Holkar Cricket Stadium',
        'M Chinnaswamy Stadium', 'Wankhede Stadium', 'Eden Gardens',
        'Feroz Shah Kotla',
        'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Green Park', 'Punjab Cricket Association Stadium, Mohali',
        'Sawai Mansingh Stadium', 'MA Chidambaram Stadium, Chepauk',
        'Dr DY Patil Sports Academy', 'Newlands', "St George's Park",
        'Kingsmead', 'SuperSport Park', 'Buffalo Park',
        'New Wanderers Stadium', 'De Beers Diamond Oval',
        'OUTsurance Oval', 'Brabourne Stadium',
        'Sardar Patel Stadium, Motera', 'Barabati Stadium',
        'Vidarbha Cricket Association Stadium, Jamtha',
        'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Subrata Roy Sahara Stadium',
        'Shaheed Veer Narayan Singh International Stadium',
        'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
        'Sharjah Cricket Stadium'],
        np.arange(0, 34), inplace=True)
        prediction = model.predict(input_variables)
        prediction = pd.DataFrame(prediction, columns=['Winners'])
        prediction = prediction["Winners"].map({0:'Sunrisers Hyderabad', 1:'Mumbai Indians', 2:'Gujarat Lions',
                      3:'Rising Pune Supergiant', 4:'Royal Challengers Bangalore',
                      5:'Kolkata Knight Riders', 6:'Delhi Capitals', 7:'Kings XI Punjab',
                      8:'Chennai Super Kings', 9:'Rajasthan Royals', 10:'Deccan Chargers',
                      11:'Kochi Tuskers Kerala', 12:'Pune Warriors', 13:'Rising Pune Supergiants'})
        return render_template('main.html', original_input={'city':city, 'Home':Home, 'Away':Away, 'toss_winner':toss_winner, 'toss_decision':toss_decision,
                                     'venue':venue},
                                    result=prediction[0],
                                    )
# Continuing from your last snippet
@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    if request.method == 'POST':    

        venue = request.form['venue']
        if venue == 'Eden Gardens':
            temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'M Chinnaswamy Stadium':
            temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Feroz Shah Kotla':
            temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Wankhede Stadium':
            temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'MA Chidambaram Stadium, Chepauk':
            temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Punjab Cricket Association Stadium, Mohali':
            temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Sawai Mansingh Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Rajiv Gandhi International Stadium, Uppal':
            temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Sardar Patel Stadium, Motera':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Kingsmead':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Brabourne Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'SuperSport Park':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Dubai International Cricket Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Himachal Pradesh Cricket Association Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Sheikh Zayed Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Sharjah Cricket Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == "St George's Park":
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'JSCA International Stadium Complex':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'New Wanderers Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif venue == 'Shaheed Veer Narayan Singh International Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif venue == 'Dr DY Patil Sports Academy':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif venue == 'Maharashtra Cricket Association Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif venue == 'Newlands':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif venue == 'Barabati Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif venue == 'Holkar Cricket Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif venue == 'Buffalo Park':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif venue == 'OUTsurance Oval':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif venue == 'De Beers Diamond Oval':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif venue == 'Subrata Roy Sahara Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
            batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]

        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
        data = np.array([temp_array])
        filename = 'ipl_score_prediction_lr_model.pkl'
        regressor = pickle.load(open(filename, 'rb'))
        my_prediction = int(regressor.predict(data)[0])          
    return render_template('result.html', lower_limit=my_prediction-10, upper_limit=my_prediction+5)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


