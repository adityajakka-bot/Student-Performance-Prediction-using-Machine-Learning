import os, joblib, pandas as pd
MODEL_PATH="models/best_model.joblib"
def predict_student(study_hours,attendance_percent,assignment_score,sleep_hours,previous_score,participation_score):
    model=joblib.load(MODEL_PATH)
    x=pd.DataFrame([{"study_hours":study_hours,"attendance_percent":attendance_percent,
    "assignment_score":assignment_score,"sleep_hours":sleep_hours,"previous_score":previous_score,
    "participation_score":participation_score}])
    return float(max(0,min(100,model.predict(x)[0])))
if __name__=="__main__":
    print(f"Predicted final score: {predict_student(6,85,78,7,72,8):.2f}")
