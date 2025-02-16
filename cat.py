import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load category details (Book1.csv)
def load_category_info():
    df = pd.read_csv("Book1.csv")
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"case_category", "description", "next_step"}
    if not required_columns.issubset(df.columns):
        raise KeyError("Book1.csv must contain 'case_category', 'description', and 'next_step' columns.")
    
    return df


# Get category details from Book1.csv
def get_case_details(predicted_category, category_info_df):
    details = category_info_df[category_info_df["case_category"] == predicted_category]

    if details.empty:
        return {"description": "No data available", "documents": [], "next_steps": "No information available"}

    description = details["description"].values[0]
    next_steps_raw = details["next_step"].values[0]
    
    split_info = next_steps_raw.split(";")
    documents = split_info[:-1] if len(split_info) > 1 else []
    next_steps = split_info[-1] if len(split_info) > 0 else "No next steps provided"

    return {"description": description, "documents": documents, "next_steps": next_steps}


# Predict the case category
def predict_case_category(model, vectorizer, facts):
    input_vectorized = vectorizer.transform([facts])
    predicted_category = model.predict(input_vectorized)[0]
    return predicted_category


# Indian Kanoon category links
category_links = {
    "Criminal Law": "https://www.indiankanoon.org/search/?formInput=criminal+law",
    "Civil Law": "https://www.indiankanoon.org/search/?formInput=civil+law",
    "Family Law": "https://www.indiankanoon.org/search/?formInput=family+law",
    "Property Law": "https://www.indiankanoon.org/search/?formInput=property+law",
    "Consumer Law": "https://www.indiankanoon.org/search/?formInput=consumer+law",
    "Corporate Law": "https://www.indiankanoon.org/search/?formInput=corporate+law",
    "Cyber Law": "https://www.indiankanoon.org/search/?formInput=cyber+law",
}


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

        # 🔗 **Direct link to Indian Kanoon for reading similar cases**
        if predicted_category in category_links:
            case_url = category_links[predicted_category]
            st.markdown(
                f'<a href="{case_url}" target="_blank" style="text-decoration: none;">'
                '<button style="background-color: #ff5733; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer;">'
                '📖 Read Similar Cases on Indian Kanoon'
                '</button></a>',
                unsafe_allow_html=True
            )

        # 🔮 **Button for Case Winning Prediction App**
        st.markdown(
            '<a href="https://svncnve9h84cxcr2tzeqhd.streamlit.app/" target="_blank" style="text-decoration: none;">'
            '<button style="background-color: #008CBA; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer;">'
            '🔮 Predict Case Winning Probability'
            '</button></a>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
