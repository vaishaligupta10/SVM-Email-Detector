import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    "Win a Free iPhone now",
    "Meeting at 11:00 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Please find the attached report",
    "Limited offer buy now",
    "Urgent offer expires today",
    "Schedule the meeting for Monday",
    "You have won a cash prize",
    "Monthly performance report attached",
    "Exclusive deal just for you"
]

labels = [1,0,1,0,1,0,1,1,0,1,0,1]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Spam Email Classifier")

st.sidebar.header("Model Information")
st.sidebar.write(f"Improved Model Accuracy: **{accuracy:.2f}**")

new_email = st.text_area("Enter a new email message:")

if st.button("Classify"):
    if new_email.strip():
        new_email_vector = vectorizer.transform([new_email])
        prediction = svm_model.predict(new_email_vector)
        if prediction[0] == 1:
            st.error("Result: Spam Email")
        else:
            st.success("Result: Not Spam Email")
    else:
        st.warning("Please enter an email message to classify.")
