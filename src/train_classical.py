from load_data import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

train_df, test_df = load_data()

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2),stop_words='english')

x_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']

log_reg_model = LogisticRegression(max_iter=2000)
log_reg_model.fit(x_train, y_train)

svm_model = LinearSVC()
svm_model.fit(x_train, y_train)


joblib.dump(log_reg_model, 'artifacts/model_lr.pkl')
joblib.dump(svm_model, 'artifacts/model_svm.pkl')
joblib.dump(vectorizer,'artifacts/tfidf.pkl')