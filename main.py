from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
import difflib

app = Flask(__name__)

sys_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv('description.csv')
medication = pd.read_csv("medications.csv")
diets = pd.read_csv('diets.csv')

svc = pickle.load(open("SVC.pkl",'rb'))

symptoms_list = {'itching': 0,'skin_rash': 1,'nodal_skin_eruptions': 2,'continuous_sneezing': 3,'shivering': 4,'chills': 5,'joint_pain': 6,'stomach_pain': 7,'acidity': 8,'ulcers_on_tongue': 9,'muscle_wasting': 10,'vomiting': 11,'burning_micturition': 12,'spotting_ urination': 13,'fatigue': 14,'weight_gain': 15,'anxiety': 16,'cold_hands_and_feets': 17,'mood_swings': 18,'weight_loss': 19,'restlessness': 20,'lethargy': 21,'patches_in_throat': 22,'irregular_sugar_level': 23,'cough': 24,'high_fever': 25,'sunken_eyes': 26,'breathlessness': 27,'sweating': 28,'dehydration': 29,'indigestion': 30,'headache': 31,'yellowish_skin': 32,'dark_urine': 33,'nausea': 34,'loss_of_appetite': 35,'pain_behind_the_eyes': 36,'back_pain': 37,'constipation': 38,'abdominal_pain': 39,'diarrhoea': 40,'mild_fever': 41,'yellow_urine': 42,'yellowing_of_eyes': 43,'acute_liver_failure': 44,'fluid_overload': 45,'swelling_of_stomach': 46,'swelled_lymph_nodes': 47,'malaise': 48,'blurred_and_distorted_vision': 49,'phlegm': 50,'throat_irritation': 51,'redness_of_eyes': 52,'sinus_pressure': 53,'runny_nose': 54,'congestion': 55,'chest_pain': 56,'weakness_in_limbs': 57,'fast_heart_rate': 58,'pain_during_bowel_movements': 59,'pain_in_anal_region': 60,'bloody_stool': 61,'irritation_in_anus': 62,'neck_pain': 63,'dizziness': 64,'cramps': 65,'bruising': 66,'obesity': 67,'swollen_legs': 68,'swollen_blood_vessels': 69,'puffy_face_and_eyes': 70,'enlarged_thyroid': 71,'brittle_nails': 72,'swollen_extremeties': 73,'excessive_hunger': 74,'extra_marital_contacts': 75,'drying_and_tingling_lips': 76,'slurred_speech': 77,'knee_pain': 78,'hip_joint_pain': 79,'muscle_weakness': 80,'stiff_neck': 81,'swelling_joints': 82,'movement_stiffness': 83,'spinning_movements': 84,'loss_of_balance': 85,'unsteadiness': 86,'weakness_of_one_body_side': 87,'loss_of_smell': 88,'bladder_discomfort': 89,'foul_smell_of urine': 90,'continuous_feel_of_urine': 91,'passage_of_gases': 92,'internal_itching': 93,'toxic_look_(typhos)': 94,'depression': 95,'irritability': 96,'muscle_pain': 97,'altered_sensorium': 98,'red_spots_over_body': 99,'belly_pain': 100,'abnormal_menstruation': 101,'dischromic _patches': 102,'watering_from_eyes': 103,'increased_appetite': 104,'polyuria': 105,'family_history': 106,'mucoid_sputum': 107,'rusty_sputum': 108,'lack_of_concentration': 109,'visual_disturbances': 110,'receiving_blood_transfusion': 111,'receiving_unsterile_injections': 112,'coma': 113,'stomach_bleeding': 114,'distention_of_abdomen': 115,'history_of_alcohol_consumption': 116,'fluid_overload.1': 117,'blood_in_sputum': 118,'prominent_veins_on_calf': 119,'palpitations': 120,'painful_walking': 121,'pus_filled_pimples': 122,'blackheads': 123,'scurring': 124,'skin_peeling': 125,'silver_like_dusting': 126,'small_dents_in_nails': 127,'inflammatory_nails': 128,'blister': 129,'red_sore_around_nose': 130,'yellow_crust_ooze': 131}
diseases_list = {0: ['(vertigo) Paroymsal  Positional Vertigo'],1: ['AIDS'],2: ['Acne'],3: ['Alcoholic hepatitis'],4: ['Allergy'],5: ['Arthritis'],6: ['Bronchial Asthma'],7: ['Cervical spondylosis'],8: ['Chicken pox'],9: ['Chronic cholestasis'],10: ['Common Cold'],11: ['Dengue'],12: ['Diabetes '],13: ['Dimorphic hemmorhoids(piles)'],14: ['Drug Reaction'],15: ['Fungal infection'],16: ['GERD'],17: ['Gastroenteritis'],18: ['Heart attack'],19: ['Hepatitis B'],20: ['Hepatitis C'],21: ['Hepatitis D'],22: ['Hepatitis E'],23: ['Hypertension '],24: ['Hyperthyroidism'],25: ['Hypoglycemia'],26: ['Hypothyroidism'],27: ['Impetigo'],28: ['Jaundice'],29: ['Malaria'],30: ['Migraine'],31: ['Osteoarthristis'],32: ['Paralysis (brain hemorrhage)'],33: ['Peptic ulcer diseae'],34: ['Pneumonia'],35: ['Psoriasis'],36: ['Tuberculosis'],37: ['Typhoid'],38: ['Urinary tract infection'],39: ['Varicose veins'],40: ['hepatitis A']}

new_list = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue',
'muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings',
'weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes',
'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
'runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine',
'passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
'belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples',
'blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

def getPredictedValue(patientSymptoms):
    input_vec = np.zeros(len(symptoms_list))
    for item in patientSymptoms:
        input_vec[symptoms_list[item]] = 1
    return diseases_list[svc.predict([input_vec])[0]]



def allPrediction(disease):
    descr = description[description['Disease']==disease]['Description']
    descr = (' ').join([w for w in descr])
    
    prec = precautions[precautions['Disease']==disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    prec = [pre for pre in prec.values]
    
    med = medication[medication['Disease']==disease]['Medication']
    med = [med for med in med.values]
    
    die = diets[diets['Disease'] == disease]['Diet']
    die = [die for die in die.values]
    
    wrkout = workout[workout['disease']==disease]['workout']
    # wrkout = [wrk for wrk in wrkout.values]
    
    return descr , prec , med , die , wrkout
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method =='POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(",")]
        # user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
        user_symptom = []
        for i in user_symptoms:
            add = difflib.get_close_matches(i, new_list)
            if len(add)==0:
                continue
            else:
                user_symptom.append(add[0])
        print(user_symptom)
        predicted_disease = getPredictedValue(user_symptom)
        descr , prec , medi , die , wrkout = allPrediction(predicted_disease[0])
        
        new_med = medi[0][1:-1].split(",")
        new_diet = die[0][1:-1].split(",")
        return render_template('index.html',predicted_disease=predicted_disease[0],dis_des = descr,dis_pre = prec[0],dis_med = new_med,dis_diet = new_diet,dis_work=wrkout)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


if __name__ == "__main__":
    app.run(debug=True)