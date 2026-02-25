import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """
You are Dr.Bot, a helpful medical AI assistant.

Rules:
- You are NOT a real doctor.
- Always include a medical disclaimer.
- If symptoms are severe (chest pain, breathing issues, bleeding, unconsciousness), advise emergency immediately.
- Suggest general OTC medicines only.
- Never give prescription drugs dosage.
- Encourage consulting a licensed doctor.
"""

# Streamlit UI arom bhabe banate hoyee
st.set_page_config(page_title="Dr.Bot - Medical AI", page_icon="🩺")

st.title("🩺 Dr.Bot - AI Medical Assistant") # website er title
st.write("Describe your symptoms and click Generate.") # emni akta simple text lekha

user_input = st.text_area("Enter your symptoms here:") # streamlit ay text area means prompt lekhar jaiyega

if st.button("Generate Advice"): # st.button diye button baniye
    if user_input.strip() == "":
        st.warning("Please enter your symptoms.") # warning erom print kora
    else:
        full_prompt = SYSTEM_PROMPT + "\nUser: " + user_input
        
        try:
            with st.spinner("Analyzing symptoms..."): # prompt load howar somoy ata ghurbe
                response = model.generate_content(full_prompt)
                st.success("Advice Generated")
                st.write(response.text) # tor llm er response ba ans ta print erom bahbe korte hoae
        except Exception as e:
            st.error(f"Error: {e}")
            # streamlit run file_name.py application ta run korar command terminal ay lekbe