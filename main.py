from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel , Field
from fastapi.responses import RedirectResponse 

# ============================================================
# تحميل الموديل المحفوظ (Pipeline كامل: preprocessing + SVC)
# ============================================================

# 1. تحميل الموديل (تأكد من استخدام المسار النسبي المباشر للـ Docker)
try:
    model = joblib.load("models/Loan_Approval.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. إنشاء التطبيق
app = FastAPI(title="Loan Approval Prediction API")

# 3. هيكل البيانات (مع وضع أمثلة افتراضية واضحة للمستخدم لتسهيل الـ Execute)
class LoanApplication(BaseModel):
    Gender: str = Field(..., description="الجنس", examples=["Male", "Female"])
    Married: str = Field(..., description="الحالة الاجتماعية", examples=["Yes", "No"])
    Dependents: str = Field(..., description="عدد المعالين", examples=["0", "1", "2", "3+"])
    Education: str = Field(..., description="المستوى التعليمي", examples=["Graduate", "Not Graduate"])
    Self_Employed: str = Field(..., description="العمل الحر", examples=["Yes", "No"])
    ApplicantIncome: float = Field(..., description="دخل المتقدم", examples=[5000.0])
    CoapplicantIncome: float = Field(..., description="دخل الشريك المتقدم", examples=[0.0])
    LoanAmount: float = Field(..., description="مبلغ القرض بالآلاف", examples=[150.0])
    Loan_Amount_Term: float = Field(..., description="مدة القرض بالأيام", examples=[360.0])
    Credit_History: float = Field(..., description="السجل الائتماني (1 أو 0)", examples=[1.0])
    Property_Area: str = Field(..., description="منطقة العقار", examples=["Urban", "Semiurban", "Rural"])

# 4. إعادة التوجيه التلقائية لتفتح الواجهة في وجه المستخدم فوراً
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# 5. دالة التنبؤ وفحص البيانات وتفادي الأخطاء
@app.post("/predict")
def predict(application: LoanApplication):
    if model is None:
        return {"error": "Model not loaded. Check file path."}
    
    try:
        # تحويل البيانات إلى DataFrame بنفس الهيكل المنسق
        input_data = application.model_dump()  # ميزة حديثة مستقرة بديلة لـ .dict()
        input_df = pd.DataFrame([input_data])
        
        # تنفيذ التنبؤ عبر الـ Pipeline
        prediction = model.predict(input_df)[0]
        
        # التحقق من دعم الموديل لحساب الاحتمالية (تفادياً للأخطاء)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            confidence = float(proba[1] if prediction == 1 else proba[0])
        else:
            confidence = 1.0  # قيمة افتراضية إذا كان الموديل لا يدعم حساب الاحتمالية
            
        result = "Approved" if prediction == 1 else "Rejected"
        
        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        # اصطياد أي خطأ وإرجاعه بشكل مفهوم بدلاً من انهيار السيرفر
        return {"error": "Failed to process data", "details": str(e)}
