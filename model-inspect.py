import joblib

model = joblib.load("C:/Users/pc/Desktop/projects/Loan Approval/models/Loan_Approval.pkl")

print(type(model))
print("input columns" , model.feature_names_in_)