import time
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term
from problog.learning import lfi
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from patient import Patient
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)


def readDatasetFile(dataset):
    global set_diabetes
    global evidence
    global trained_model

    for index, row in data.iterrows():
        evidence_list = []
        if int(row[8]) == 1:
            final_model = "t(_)::diabetic:-"
        else:
            final_model = "t(_)::healthy:-"

        # Gender column
        is_male = row[0].lower() == 'male'
        evidence_list.append((Term("male"), is_male, None))
        final_model += "male," if is_male else "\+male,"

        # Age column
        age_user = int(float(row[1])) if row[1] != '' else 0
        is_young = age_user < 50
        evidence_list.append((Term("young"), is_young, None))
        final_model += "young," if is_young else "\+young,"

        # Hypertension
        is_hypertensive = row[2] == 1
        evidence_list.append((Term("hypertension"), is_hypertensive, None))
        final_model += "hypertension," if is_hypertensive else "\+hypertension,"

        # # Heart disease
        is_heart_disease = row[3] == 1
        evidence_list.append((Term("heartDisease"), is_heart_disease, None))
        final_model += "heartDisease," if is_heart_disease else "\+heartDisease,"

        # Smoking History
        smoking_status = row[4].lower()
        status_terms = ["never", "no info", "former", "current", "not current", "ever"]
        if smoking_status:
            for status_term in status_terms:
                evidence_list, final_model = update_evidence_and_data(smoking_status, status_term,
                                                                      evidence_list, final_model)

        # BMI
        bmi_user = float(row[5]) if row[5] != '' else 0
        is_normal_bmi = bmi_user < 25
        evidence_list.append((Term("normalBMI"), is_normal_bmi, None))
        final_model += "normalBMI," if is_normal_bmi else "\+normalBMI,"

        # HbA1c level
        hba1c_user = float(row[6]) if row[6] != '' else 0
        is_normal_hba1c = hba1c_user < 5.7
        evidence_list.append((Term("normalHbA1c"), is_normal_hba1c, None))
        final_model += "normalHbA1c," if is_normal_hba1c else "\+normalHbA1c,"

        # Blood Glucose Level
        glucose_user = int(row[7]) if row[7] != '' else 0
        is_normal_glucose = glucose_user < 140
        evidence_list.append((Term("normalGlucose"), is_normal_glucose, None))
        final_model += "normalGlucose." if is_normal_glucose else "\+normalGlucose."

        final_model = final_model.replace(",.", ".")
        set_diabetes.add(final_model)
        evidence.append(evidence_list)


def clickMethod(person):
    global trained_model
    user_evidence_infected = []
    patient_data = ""

    # Gender input
    gender_user = person.gender
    is_male = gender_user == 1 if gender_user is not None else False
    user_evidence_infected.append((Term("male"), is_male))
    patient_data += f"\nevidence(male,{is_male})."

    # Age input
    age_user = person.age
    is_young = age_user is not None and age_user < 50
    user_evidence_infected.append((Term("young"), is_young))
    patient_data += f"\nevidence(young,{is_young})."

    # Hypertension
    is_hypertensive = person.hypertension == 1
    user_evidence_infected.append((Term("hypertension"), is_hypertensive))
    patient_data += f"\nevidence(hypertension,{is_hypertensive})."

    # # Heart disease
    is_heart_disease = person.heart_disease == 1
    user_evidence_infected.append((Term("heartDisease"), is_heart_disease))
    patient_data += f"\nevidence(heartDisease,{is_heart_disease})."

    # Smoking History
    smoking_status = person.smoking_history.lower()
    status_terms = ["never", "no info", "former", "current", "not current", "ever"]
    if smoking_status:
        for status_term in status_terms:
            user_evidence_infected, patient_data = update_evidence_and_data(smoking_status, status_term,
                                                                            user_evidence_infected, patient_data,
                                                                            is_patient_data=True)
    # BMI
    bmi_user = float(person.bmi) if person.bmi != '' else 0
    is_normal_bmi = bmi_user < 25
    user_evidence_infected.append((Term("normalBMI"), is_normal_bmi))
    patient_data += f"\nevidence(normalBMI,{is_normal_bmi})."

    # HbA1c level
    hba1c_user = float(person.hba1c_level) if person.hba1c_level != '' else 0
    is_normal_hba1c = hba1c_user < 5.7
    user_evidence_infected.append((Term("normalHbA1c"), is_normal_hba1c))
    patient_data += f"\nevidence(normalHbA1c,{is_normal_hba1c})."

    # Blood Glucose Level
    glucose_user = int(person.blood_glucose) if person.blood_glucose != '' else 0
    is_normal_glucose = glucose_user < 140
    user_evidence_infected.append((Term("normalGlucose"), is_normal_glucose))
    patient_data += f"\nevidence(normalGlucose,{is_normal_glucose})."

    # Final query to find the risk prediction
    patient_data = patient_data + "\nquery(diabetic)."

    # Final query to find healthy probability
    patient_data = patient_data + "\nquery(healthy)."

    # User model is the evidences from the user inputs above
    patient_data = trained_model + patient_data

    # Evaluate the new model with the evidences with Problog
    p_usermodel = PrologString(patient_data)
    result = get_evaluatable().create_from(p_usermodel, propagate_evidence=True).evaluate()

    cnt = 0
    for query, value in result.items():
        if cnt == 0:
            prob_message = "Probability to be diabetic: " + format(value, ".4f") + "\n"
            cnt = cnt + 1
        else:
            prob_message = prob_message + "Probability to be healthy: " + format(value, ".4f") + "\n"
            cnt = 0
    print(prob_message)
    return prob_message
    # print(user_evidence_infected)
    # print(result)


def update_evidence_and_data(smoking_status, status_term, evidence_list, final_model, is_patient_data=False):
    status_boolean = smoking_status == status_term
    status_term_replaced = status_term.replace(" ", "_")
    if is_patient_data:
        evidence_list.append((Term(status_term_replaced), status_boolean))
        final_model += f"\nevidence({status_term_replaced},{str(status_boolean).lower()})."
    else:
        evidence_list.append((Term(status_term_replaced), status_boolean, None))
        final_model += status_term_replaced if status_boolean else "\\+" + status_term_replaced
        final_model += ","
    return evidence_list, final_model


def get_model_classification(person):
    prediction = clickMethod(person)
    # Now we extract the float values from the string
    lines = prediction.split("\n")
    diabetic_prob = float(lines[0].split(":")[1].strip())
    healthy_prob = float(lines[1].split(":")[1].strip())

    return 1 if diabetic_prob >= 0.5 and healthy_prob < 0.5 else 0


def get_actual_classification(row):
    return int(row[8])


# if __name__ == "__main__":
model_path = os.path.join(os.getcwd(), 'trained_model.pkl')

if os.path.exists(model_path):
    print("Trained model found, loading...")
    # Load the trained model from the file
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

# Use the trained model here if web app is not used
# patient = Patient('female', 5, 0, 1, 'never', 25.19, 6.6, 200)
# clickMethod(patient)

else:
    print("No trained model found, creating a new one...")
    # First I create a set and a list to save the csv and input user data
    set_diabetes = set()
    evidence = list()

    # Load the dataset
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    # Split the dataset into training and testing
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Function that reads from csv and creates the training model
    readDatasetFile(train_data)

    term_list = list(set_diabetes)
    term_list.sort()

    # Creating the Learning Model
    model = """"""
    model = model + "t(_)::male.\n"
    model = model + "t(_)::young.\n"
    model = model + "t(_)::hypertension.\n"
    model = model + "t(_)::heartDisease.\n"
    model = model + "t(_)::never.\n"
    model = model + "t(_)::no_info.\n"
    model = model + "t(_)::former.\n"
    model = model + "t(_)::current.\n"
    model = model + "t(_)::not_current.\n"
    model = model + "t(_)::ever.\n"
    model = model + "t(_)::normalBMI.\n"
    model = model + "t(_)::normalHbA1c.\n"
    model = model + "t(_)::normalGlucose.\n"

    for y in range(len(term_list)):
        if y != (len(term_list) - 1):
            model = model + term_list[y] + "\n"
        else:
            model = model + term_list[y]

    # print(model)

    # Evaluate the learning model
    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(model), evidence)
    trained_model = lfi_problem.get_model()
    # print(trained_model)

    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    print("Model created")

    # Convert test_data into a list of Patient objects
    test_patients = [Patient(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]) for row in
                     test_data.itertuples(index=False)]

    # Get the actual and predicted classifications
    actual_classifications = [get_actual_classification(row) for row in test_data.itertuples(index=False)]
    predicted_classifications = [get_model_classification(patient) for patient in test_patients]
    # Get the time it takes to compute the metrics
    start_time = time.time()

    # Calculate the metrics
    acc = accuracy_score(actual_classifications, predicted_classifications)
    prec = precision_score(actual_classifications, predicted_classifications)
    rec = recall_score(actual_classifications, predicted_classifications)
    f1 = f1_score(actual_classifications, predicted_classifications)


    # Write the metrics to a file
    with open('results.txt', 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        f.write(f'Precision: {prec}\n')
        f.write(f'Recall: {rec}\n')
        f.write(f'F1 Score: {f1}\n')

        # Compute and write the confusion matrix
        cm = confusion_matrix(actual_classifications, predicted_classifications)
        f.write(f'Confusion Matrix: \n{cm}\n')

        end_time = time.time()
        execution_time = (end_time - start_time) / 60.0
        f.write(f'Execution time: {execution_time} minutes\n')

@app.route('/diabetes-prediction', methods=['POST'])
def predict_diabetes():
    jsonData = request.get_json(force=True)  # forcing the interpretation of request data as JSON

    patient = Patient(jsonData['gender'], int(jsonData['age']), int(jsonData['hypertension']), int(jsonData['heart_disease']),
                      jsonData['smoking_status'], float(jsonData['bmi']), float(jsonData['hba1c']), int(jsonData['glucose']))

    prediction = clickMethod(patient)  # prediction method

    return jsonify({
        'probability': prediction
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
