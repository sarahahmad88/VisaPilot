# Viza Pilot — Germany Student Visa RAG MVP

This is the first focused MVP for **Viza Pilot**.

It supports one core use case:

**Pakistani applicant → Germany → Student Visa → ask a visa question → retrieve evidence from the supplied visa guide → generate a grounded answer with Groq.**

The repository deliberately contains **only three files**:

```text
app.py
requirements.txt
README.md
```

You do **not** need to run the app on your own computer before deploying it.

---

## 1. What this app does

The application is a simple **RAG (Retrieval-Augmented Generation)** system.

```text
Visa guide knowledge
       ↓
Sentence embeddings
       ↓
FAISS vector index
       ↓
User asks a question
       ↓
FAISS retrieves the most relevant passages
       ↓
Retrieved passages + question go to Groq
       ↓
Groq writes a grounded answer
       ↓
The app shows the answer + PDF evidence/pages
```

### Why this is useful

A normal chatbot can answer from its general model knowledge. That is risky for visa information because it may invent, mix up, or use outdated requirements.

This MVP instead instructs Groq to answer factual visa questions **only from passages retrieved from your supplied Germany visa guide**.

If the guide does not support an answer, the assistant is instructed to say that it could not verify the point from the guide.

---

## 2. Why the PDF is not a fourth repository file

You asked for exactly three files.

For that reason, the relevant student-visa information from your supplied 9-page PDF has been converted into page-labelled knowledge passages directly inside `app.py`.

That gives you this GitHub repository:

```text
viza-pilot-rag/
├── app.py
├── requirements.txt
└── README.md
```

The app therefore does **not** require the user to re-upload the Germany visa guide every time Streamlit restarts.

Later, when Viza Pilot supports many countries and visa types, the knowledge should be moved into a proper database/document ingestion pipeline rather than embedded in `app.py`.

---

## 3. Technologies used

### Streamlit
Provides the website/chat interface.

### FAISS
Stores the vector representations of the visa-guide passages and finds passages semantically related to the user's question.

### Sentence Transformers
The app uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert the visa-guide passages and user questions into numerical embeddings that FAISS can compare.

### Groq
Groq generates the natural-language answer after FAISS has retrieved the relevant visa-guide passages.

The default production model configured in the app is:

```text
openai/gpt-oss-120b
```

You can change the Groq model later without changing the Python code by setting `GROQ_MODEL` in Streamlit Secrets.

---

# 4. Deploy directly from GitHub to Streamlit

## Step 1 — Create a GitHub repository

Go to GitHub and create a new repository.

A suitable repository name is:

```text
viza-pilot-rag
```

For your early MVP, keeping the repository **Private** is reasonable.

---

## Step 2 — Upload the three files

Upload these files to the root of your GitHub repository:

```text
app.py
requirements.txt
README.md
```

Do not place them inside another folder in the repository.

The GitHub repository should look like this:

```text
viza-pilot-rag
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Step 3 — Create your Groq API key

Create an API key in the Groq Console.

**Never put the API key directly into `app.py` or GitHub.**

The key is a secret credential and should be stored only in Streamlit's Secrets area.

---

## Step 4 — Open Streamlit Community Cloud

Sign in to Streamlit Community Cloud with your GitHub account.

Choose the option to create/deploy a new app.

Select:

```text
Repository: YOUR_GITHUB_USERNAME/viza-pilot-rag
Branch: main
Main file path: app.py
```

Then create/deploy the app.

The first deployment can take several minutes because Streamlit must install FAISS and the sentence-transformer dependencies.

---

## Step 5 — Add the Groq API key

In the Streamlit application settings, open **Secrets** and add:

```toml
GROQ_API_KEY = "paste-your-real-groq-key-here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Do not add quotation marks around the key twice and do not commit this information to GitHub.

After saving the Secrets, reboot/redeploy the Streamlit app if necessary.

---

## Step 6 — Test the app

Try questions such as:

```text
What documents do I need for a German student visa?
```

```text
How much money do I need in a blocked account?
```

```text
I live in Lahore. Which mission handles my application?
```

```text
Can an HEC scholarship count as financial proof?
```

```text
What happens after I upload my documents to the Consular Services Portal?
```

For every answer, open **See the PDF evidence used**.

That is important: it lets you see whether FAISS retrieved the correct passages before Groq generated the answer.

---

# 5. What each file does

## `app.py`

This contains the entire application:

- Streamlit user interface
- visa-guide knowledge passages
- sentence-transformer embeddings
- FAISS vector search
- Groq API connection
- RAG prompt/guardrails
- chat history
- page-aware evidence display

## `requirements.txt`

This tells Streamlit which Python libraries must be installed automatically.

## `README.md`

This is the human-readable documentation you are reading now. GitHub displays it automatically on the repository homepage.

---

# 6. Important RAG safety design

Viza Pilot uses this architecture:

```text
Visa guide
   ↓
FAISS retrieval
   ↓
Relevant passages
   ↓
Groq
   ↓
User-friendly explanation
```

It intentionally does **not** use this architecture:

```text
User question
   ↓
Groq's general knowledge
   ↓
Visa requirement presented as fact
```

The second design is unsafe for a visa product because an LLM can hallucinate or use outdated information.

The system prompt therefore tells Groq:

- use retrieved PDF context for factual visa answers;
- do not invent missing requirements;
- say when something cannot be verified from the guide;
- cite the guide's page numbers;
- never guarantee visa approval;
- remind users to verify time-sensitive requirements with the responsible official authority.

---

# 7. Important limitation of the source guide

Your supplied Germany visa guide says it was compiled from German Missions in Pakistan sources, but it also states that the guide itself is an **independent informational summary**, not an official publication.

It also warns that values such as fees, blocked-account thresholds and procedures may change.

That is why the app presents the guide as the MVP knowledge base while still telling users to verify time-sensitive information before applying.

For a production version of Viza Pilot, the next step should be a versioned **Official Visa Rules Database** that records:

- country;
- visa type;
- requirement;
- official source URL;
- source authority;
- date last verified;
- effective date;
- verification status;
- prior rule version where relevant.

---

# 8. What this MVP intentionally does NOT do yet

This version does not yet include:

- user registration/login;
- persistent applicant profiles;
- document uploads such as passports or bank statements;
- OCR/document extraction;
- cross-document consistency checks;
- personalized applicant checklists;
- readiness scoring;
- payment/subscription handling;
- appointment monitoring;
- notification alerts;
- Schengen 90/180 tracking;
- multi-country visa databases.

That is deliberate.

The purpose of this version is to prove the first AI capability:

> **Can Viza Pilot reliably answer German student-visa questions using a controlled visa knowledge source?**

Once this works reliably, it becomes the knowledge/AI layer for the larger Viza Pilot MVP.

---

# 9. Your next development milestone

After this RAG assistant is working on Streamlit, the most sensible next milestone is:

```text
Applicant profile interview
        ↓
Structured applicant information
        ↓
Personalized requirements/checklist
        ↓
RAG explanations for each requirement
```

After that, add document upload and document analysis.

This keeps development aligned with the Viza Pilot PRD without attempting to build the entire product at once.
