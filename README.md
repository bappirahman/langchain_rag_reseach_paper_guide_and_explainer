# Research Paper Guide and Explainer

This project is a research paper guide and explainer web application built using Streamlit, Langchain, and Hugging Face. It allows users to input an Arxiv query number of a research paper and ask a question about that specific research paper. The application uses the Langchain library to load the data from Arxiv, split the documents into chunks, and compute the vector embeddings of the documents. The application then uses the Hugging Face Embeddings model to answer the user's question based on the context provided in the research papers.


Live demo: https://arxivragapp.streamlit.app/



## Installation

To run this project, you need to have Python and the required packages installed. Follow these steps to install the required packages:

1. Clone the repository:

  ```bash
  git clone https://github.com/ArXiv-ML/research-paper-guide-and-explainer.git
  ```

2. Install the required packages:

  ```bash
  cd research-paper-guide-and-explainer
  pip install -r requirements.txt
  ```

**Note:** You need to create a .env file in the root directory of the project and add your Groq API key as `GROQ_API_KEY`.

## Usage

To use this project, simply enter an Arxiv query number of a research paper and ask a question about that specific research paper. The application will then provide an answer based on the context provided in the research papers.


## License

This project is licensed under the MIT License.



## Disclaimer

This project is in no way affiliated with ArXiv

