class Patient:
    def __init__(self, gender, age, hypertension, heart_disease,
                 smoking_history, bmi, hba1c_level, blood_glucose):
        self.age = age
        self.gender = gender
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.smoking_history = smoking_history
        self.bmi = bmi
        self.hba1c_level = hba1c_level
        self.blood_glucose = blood_glucose