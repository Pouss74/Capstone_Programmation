import streamlit as st
from datetime import datetime

def main():
    # Set the page config to add a favicon and modify the layout
    st.set_page_config(page_title="Crypto Medium", page_icon="🔮", layout="wide")

    # Display the app title and a small description using markdown
    st.title('Crypto Medium 🔮')
    st.markdown("Select a cryptocurrency and a date to get the prediction.")

    # Create two columns for inputs: one for cryptocurrency selection and one for date selection
    col1, col2 = st.columns(2)

    # Cryptocurrency selection in the first column
    with col1:
        coin_type = st.radio("Choose Cryptocurrency:", ('Bitcoin', 'Ethereum'))

    # Date selection in the second column
    with col2:
        prevision_date = st.date_input("Choose a Date for Prediction:", min_value=datetime.today())

    # A button to trigger the prediction process
    if st.button('Get Prediction'):
        st.write(f"Prediction for {coin_type} on {prevision_date.strftime('%Y-%m-%d')} will be displayed here.")
        # Note: Integrate your prediction logic here to show actual results

if __name__ == "__main__":
    main()
