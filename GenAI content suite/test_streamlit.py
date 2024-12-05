import os
import streamlit as st
import openai
import pinecone

from openai import OpenAI

# Assuming you have the secrets manager setup, otherwise replace with your own method for securing API keys
os.environ["OPENAI_API_KEY"] = "sk-"
client = OpenAI()

pinecone.init(
	api_key='KEY_CODE',
	environment='us-west1-gcp-free'
)
index = pinecone.Index('movie-recommendation')

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Title of the application
st.title('OpenAI API Applications')

# Sidebar for navigation
st.sidebar.title("Applications")
applications = ["Article Generation", "Generate Image", "Movie Recommendation"]
application_choice = st.sidebar.radio("Choose an Application", applications)

def article_generation(topic, additional_pointers):
    prompt = f"""
    You are a copy writer with years of experience writing impactful blog that converge and help elevate brands.
    Your task is to write a blog on any topic system provides you with. Make sure to write in a format that works for Medium.
    Each blog should be separated into segments that have titles and subtitles. Each paragraph should be three sentences long.

    Topic: {topic}
    Additiona pointers: {additional_pointers}
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=1,
        max_tokens=700,
    )

    return response.choices[0].text.strip()

def generate_image(prompt, number_of_images=1):
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",
        n=number_of_images,
    )

    return response

# Main application logic
def main():
    if application_choice == "Article Generation":
        st.header("Text Completion with GPT-3")
        st.write("Input some text and get a completion.")
        input_text = st.text_area("Enter text here:")
        additional_pointers = st.text_area("Enter additional pointers here:")
        
        if st.button("Complete Text"):
            with st.spinner('Generating...'):
                completion = article_generation(input_text, additional_pointers)
                st.text_area("Generated artice:", value=completion, height=200)

    elif application_choice == "Generate Image":
        st.header("Image Generation with DALL-E")
        st.write("Input some text and generate an image.")
        input_text = st.text_area("Enter text for image generation:")

        number_of_images = st.slider("Choose the number of images to generate", 1, 5, 1) 
        if st.button("Generate Image"):
            
            outputs = generate_image(input_text, number_of_images)
            for output in outputs.data:
                st.image(output.url)

    elif application_choice == "Movie Recommendation":
        st.header("Movie Recommendation with GPT")
        st.write("Input a movie description and get a recommendation.")

        input_text = st.text_area("Enter movie description:")

        if st.button("Get movies"):
            with st.spinner('Generating...'):
                user_vector = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=input_text)

                user_vector = user_vector.data[0].embedding
                matches = index.query(
                    user_vector,
                    top_k=10,
                    include_metadata=True)

                for match in matches:
                    st.write(match['metadata']['title'])
        

# Run the main function
if __name__ == "__main__":
    main()
