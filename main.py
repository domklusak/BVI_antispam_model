import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Load dataset
data = pd.read_csv('spam.csv', usecols=[0,1], names=['label', 'text'], skiprows=1)
print(data.head())

# split to train/test
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# prediction of test model
y_pred = model.predict(X_test_vec)

# Evaluation of model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('Skutočná trieda')
plt.xlabel('Predpovedaná trieda')
plt.title('Confusion Matrix')
plt.show()

y_scores = model.predict_proba(X_test_vec)[:, 1]

# calculating roc curve and auc score
fpr, tpr, threshold = roc_curve(y_test, y_scores, pos_label='spam')
auc = roc_auc_score(y_test, y_scores)

# roc curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC krivka (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falošne pozitívna miera')
plt.ylabel('Pravdivo pozitívna miera')
plt.title('ROC krivka')
plt.legend(loc="lower right")
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_weights = model.feature_log_prob_[1, :]

# top words for spam
sorted_spam_weights = sorted(zip(spam_weights, feature_names), reverse=True)
top_spam_words = sorted_spam_weights[:20]

# vizualization of top words
plt.figure(figsize=(10, 8))
plt.barh([word for _, word in top_spam_words], [weight for weight, _ in top_spam_words])
plt.xlabel('Váha')
plt.title('Najpoužívanejšie slova pre spam')
plt.gca().invert_yaxis()
plt.show()
