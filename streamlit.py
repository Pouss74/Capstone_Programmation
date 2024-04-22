import streamlit as st
from datetime import date

def main():
    st.set_page_config(page_title="Crypto Medium", page_icon="🔮", layout="wide")

    st.title('Crypto Medium 🔮')
    st.subheader('Explore Cryptocurrency Predictions')

    # UI for selecting cryptocurrency
    coin = st.radio("Select Cryptocurrency:", ['Bitcoin', 'Ethereum'], index=0)

    # UI for selecting date
    prediction_date = st.date_input("Select Prediction Date:", min_value=date.today())

    st.markdown(f"You have selected **{coin}** and the date **{prediction_date}**.")
    st.write("🔮 Prediction results will be displayed here after integrating predictive analysis.")

if __name__ == "__main__":
    main()

