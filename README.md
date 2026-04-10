# 🎓 METU IE Summer Practice Chatbot

An intelligent chatbot that assists METU Industrial Engineering students with questions about IE 300 and IE 400 Summer Practice procedures, documents, deadlines, and report guidelines.

## 🏗️ Architecture

- **Frontend:** Streamlit (Python)
- **LLM:** Google Gemini 2.0 Flash (free tier)
- **Embeddings:** Gemini `text-embedding-004`
- **Vector Search:** NumPy cosine similarity (no external DB needed)
- **Knowledge Base:** 13 curated text files from [sp-ie.metu.edu.tr](https://sp-ie.metu.edu.tr/en)

### How it works (RAG Pipeline)

1. **Data Collection:** All pages from the SP website are scraped and cleaned into structured text files.
2. **Chunking:** Text files are split into ~800 character overlapping chunks.
3. **Embedding:** Each chunk is embedded using Gemini's embedding model.
4. **Retrieval:** When a user asks a question, the question is embedded and the top-5 most similar chunks are retrieved via cosine similarity.
5. **Generation:** The retrieved chunks are sent as context to Gemini 2.0 Flash along with the question, which generates a grounded answer.
6. **Out-of-scope handling:** The system prompt instructs the model to politely decline questions outside the SP domain.

## 🚀 Deploy to Streamlit Cloud (Recommended)

1. **Fork or push this repo to GitHub**

2. **Go to [share.streamlit.io](https://share.streamlit.io)**
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and `app.py`

3. **Add your API key:**
   - In the app settings → "Secrets", paste:
     ```
     GEMINI_API_KEY = "your-actual-key-here"
     ```
   - Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

4. **Click Deploy.** Your app will be live at `https://your-app.streamlit.app`

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd metu-ie-sp-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export GEMINI_API_KEY="your-key-here"

# 4. Run
streamlit run app.py
```

## 🧪 Testing Guide — Sample Queries

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 1 | "What are the prerequisites for IE 300?" | Lists IE 102, IE 251, IE 265, IE 241, OHS 101, and one of IE 266/IE 252 |
| 2 | "SGK sigortası için ne zaman başvurmalıyım?" | Responds in Turkish: 2-3 hafta önce, OCW üzerinden |
| 3 | "What companies offer summer practice in Ankara?" | Lists relevant companies from SP Opportunities |
| 4 | "How is the IE 400 report graded?" | Explains 200 pts (questions) + 100 pts (problem/project), section minimums |
| 5 | "What is the weather in Ankara?" | Politely declines — out of scope |

## 📁 Project Structure

```
metu-ie-sp-chatbot/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml           # UI theme
│   └── secrets.toml.example  # API key template
├── knowledge_base/           # 13 curated text files
│   ├── 01_general_information.txt
│   ├── 02_steps_to_follow.txt
│   ├── 03_faq.txt
│   ├── 04_sp_committee.txt
│   ├── 05_home_and_announcements.txt
│   ├── 06_custom_faq.txt
│   ├── 07_documents_forms.txt
│   ├── 08_sp_opportunities.txt
│   ├── 09_previous_sp_opportunities.txt
│   ├── 10_ie300_introductory_session.txt
│   ├── 11_ie400_introductory_session.txt
│   ├── 12_ie300_manual.txt
│   └── 13_ie400_manufacturing_manual.txt
└── README.md
```

## 📝 Knowledge Base Sources

All data is sourced from the official METU IE Summer Practice website: https://sp-ie.metu.edu.tr/en

- General Information, Steps to Follow, FAQ, SP Committee, Documents/Forms
- SP Opportunities & Previous SP Opportunities
- IE 300 & IE 400 Introductory Session slides (2025-2026)
- IE 300 Summer Practice Manual
- IE 400 Summer Practice Manufacturing Manual
- Custom FAQ (16 student-oriented questions compiled from official content)
