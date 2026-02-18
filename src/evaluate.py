from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from load_data import load_data
import joblib

model_lr = joblib.load('artifacts/model_lr.pkl')
model_svm = joblib.load('artifacts/model_svm.pkl')
vectorizer = joblib.load('artifacts/tfidf.pkl')

train_df, test_df = load_data()

x_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

y_pred_lr = model_lr.predict(x_test)
y_pred_svm = model_svm.predict(x_test)


print(classification_report(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_svm))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
plt.show()
