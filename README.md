# HealthGPT - A Conversational AI for Medical Information Access

## Overview:

This project, created for the GenAI Genesis Hackathon 2024, empowers patients with a user-friendly interface to access medical knowledge from a curated collection of medical textbooks like a digital doctor! ‍⚕️ By leveraging the power of large language models (LLMs) and natural language processing (NLP) techniques, HealthGPT facilitates interactive conversations where users can describe their symptoms and medical history, receiving informative responses based on the processed information ℹ️.

## Features:

**Streamlit Integration:** The project utilizes Streamlit for a streamlined web-based application, enabling users to interact with HealthGPT in their browser.

**Conversational Interface:** ️ Users can engage with HealthGPT through a chat-like interface, posing questions about their health concerns in a natural language format.

**Medical Text Processing:** HealthGPT processes medical textbooks using NLP techniques, extracting relevant information and organizing it for efficient retrieval.

**Conversational Retrieval:** When a user asks a question, HealthGPT employs an LLM to search its knowledge base, retrieving the most pertinent information from the processed medical texts.

**Chat History:** The conversation history is maintained, allowing users to refer back to previous interactions and providing context for ongoing discussions.

## Getting Started:

**Prerequisites:** Ensure you have Python (version 3.11 recommended) and the required libraries installed (streamlit, langchain, etc.) You can install them using "pip install -r requirements.txt".

**Data Preparation:** Place your medical textbooks in the data directory, ensuring they are in PDF format.

**Download LLM Model:** Download the quantized version of the Llama-2 GPT model from Hugging Face: https://huggingface.co/docs/transformers/main/en/model_doc/llama (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin) and place it in the same directory as your project files (e.g., alongside app.py). This model is optimized for faster inference on your local machine.

*Optional:* Hugging Face API Token

For advanced functionalities or future updates that might leverage Hugging Face APIs, you can optionally obtain a Hugging Face API token from Hugging Face: https://huggingface.co/docs/hub/en/security-tokens and create a .env file in your project directory. Paste the API token inside the .env file with the following key-value pair:

HUGGINGFACE_API_TOKEN="your_hugging_face_api_token"

## Run the App:

Navigate to the project directory in your terminal.

Execute "streamlit run app.py". This will launch the HealthGPT web application in your default browser.

## Usage:

Open the HealthGPT web application in your browser.

In the chat interface, type your questions about medical symptoms, conditions, or general health inquiries.

HealthGPT will process your query, retrieve relevant information from its knowledge base, and provide a response.

You can continue the conversation by asking further questions to refine your search or delve deeper into specific topics.

## Disclaimer:

**Important Note** - HealthGPT is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a licensed physician for diagnosis, treatment, and personalized medical guidance.

## Additional Notes:

This is a research project and is under development.

The accuracy of the responses may vary depending on the quality and comprehensiveness of the medical textbooks used.

It's crucial to consult with a healthcare professional for any medical concerns ‍⚕️.

## Future Enhancements:

We envision further enhancements to HealthGPT, such as:

Incorporating additional medical resources beyond textbooks, such as research articles and patient information leaflets.

Implementing symptom checker functionalities to provide preliminary insights (emphasizing the need for professional consultation).
