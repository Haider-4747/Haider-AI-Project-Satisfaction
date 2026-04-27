import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. تحميل البيانات
df = pd.read_csv('ai_job_impact.csv')

# 2. تنظيف بسيط: تحويل البيانات النصية لأرقام عشان يفهمها الذكاء الاصطناعي
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})

# 3. تحديد وش نبي نتنبأ فيه (الرضا الوظيفي) وبناء على ايش (العمر والإنتاجية)
X = df[['Age', 'Productivity_Change_%']] # المدخلات
y = df['Job_Satisfaction'] # الهدف اللي نبي نتنبأ فيه

# 4. تقسيم البيانات (جزء للتدريب وجزء للاختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب نموذج الذكاء الاصطناعي (Random Forest)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 6. تجربة التنبؤ
score = model.score(X_test, y_test)
print(f"✅ تم تدريب النموذج بنجاح!")
print(f"📊 دقة النموذج في التنبؤ: {score:.2f}")

# مثال تنبؤ لموظف جديد عمره 30 وتغير إنتاجيته 10%
sample_pred = model.predict([[30, 10]])
print(f"🔮 التنبؤ لموظف (عمر 30، إنتاجية 10%): مستوى الرضا المتوقع هو {sample_pred[0]:.2f}")