import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = './trained_model'


@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


def format_code(unformatted_code, max_length=512):
    input_ids = tokenizer.encode(
        unformatted_code,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )

    formatted_code = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return formatted_code.strip()


st.title("Code Formatter")
st.write("Enter unformatted code and receive formatted code as per learned conventions.")

code_input = st.text_area("Unformatted Code", height=200)

if st.button("Format Code"):
    if code_input.strip() == "":
        st.warning("Please enter some code.")
    else:
        formatted_code = format_code(code_input)
        st.subheader("Formatted Code")
        st.code(formatted_code, language='python')
