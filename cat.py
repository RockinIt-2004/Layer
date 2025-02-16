import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the main dataset (justice.csv)
def load_data():
    df = pd.read_csv("justice.csv")  # Your main dataset
    df.rename(columns={"facts": "facts", "issue_area": "case_category"}, inplace=True)

    # Ensure required columns exist
    if "facts" not in df.columns or "case_category" not in df.columns:
        raise KeyError("Required columns 'facts' and 'case_category' are missing from justice.csv")
    
    # Drop missing values
    df.dropna(subset=["facts", "case_category"], inplace=True)
    return df


# Load category details (Book1.csv)
def load_category_info():
    df = pd.read_csv("Book1.csv")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    required_columns = {"case_category", "description", "next_step"}
    if not required_columns.issubset(df.columns):
        raise KeyError("Book1.csv must contain 'case_category', 'description', and 'next_step' columns.")
    
    return df


# Train the model
def train_category_model(df):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["facts"])
    y = df["case_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the model and vectorizer
    joblib.dump(model, "case_category_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return model, vectorizer


# Get category details from Book1.csv
def get_case_details(predicted_category, category_info_df):
    """Fetch details for the predicted case category."""
    
    # Ensure required columns exist
    required_columns = {"case_category", "description", "next_step"}
    if not required_columns.issubset(category_info_df.columns):
        return {"description": "Missing required data", "documents": [], "next_steps": "No information available"}
    
    # Filter by case category
    details = category_info_df[category_info_df["case_category"] == predicted_category]

    if details.empty:
        return {"description": "No data available", "documents": [], "next_steps": "No information available"}

    # Extract details
    description = details["description"].values[0]
    next_steps_raw = details["next_step"].values[0]

    # Split required documents and steps
    split_info = next_steps_raw.split(";")
    documents = split_info[:-1] if len(split_info) > 1 else []
    next_steps = split_info[-1] if len(split_info) > 0 else "No next steps provided"

    return {"description": description, "documents": documents, "next_steps": next_steps}


# Predict the case category
def predict_case_category(model, vectorizer, facts):
    """Predict the legal category of a case based on input facts."""
    input_vectorized = vectorizer.transform([facts])
    predicted_category = model.predict(input_vectorized)[0]

    return predicted_category


# Streamlit App
def main():
    st.title("Legal Case Classification System")
    
    # Load category details
    category_info_df = load_category_info()

    # Load trained model and vectorizer
    try:
        model = joblib.load("case_category_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except FileNotFoundError:
        st.error("Model files not found! Please train the model first.")
        return

    # User input
    user_input = st.text_area("Enter the case details:", "")
    
    if st.button("Predict Case Category"):
        if not user_input.strip():
            st.error("Please enter case details before predicting.")
            return

        # Predict category
        predicted_category = predict_case_category(model, vectorizer, user_input)

        # Fetch case details
        case_info = get_case_details(predicted_category, category_info_df)

        # Display results
        st.subheader(f"Predicted Case Category: {predicted_category}")
        st.write(f"**Description:** {case_info['description']}")

        st.write("**Required Documents:**")
        if case_info["documents"]:
            for doc in case_info["documents"]:
                st.write(f"- {doc}")
        else:
            st.write("No specific documents listed.")

        st.write(f"**Next Steps:** {case_info['next_steps']}")

if __name__ == "__main__":
    main()
