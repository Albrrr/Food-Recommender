# Food-Recommender
A food recommendation website that is powered by AI to provide tailored restaurants to users based on their input.


## Project-Structure



```

food-recommendation-rag/

├── backend/
│   ├── faiss_store/
│   │   ├── index.faiss
│   │   └── index.pkl
│   │ 
│   ├── app.py
│   ├── model.py
│   ├── rag.py
│   ├── vectorstore.py
│   ├── schemas.py
│   ├── colab_run.py
│   ├── DATA.json
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── README.md
│
├── .gitignore
└── README.md

```

## Instructions
Backend
- Open terminal(Powershell)
- Set the HF_TOKEN with export HF_TOKEN=hf_......
- Navigate to backend/
- Run the server with uvicorn app:app --host 0.0.0.0 --port 8000
- ngrok http 8000

### Hosting the backend on Google Colab (recommended for GPU)

- **Create a Colab notebook** and upload / clone this repo into the Colab runtime.
- **Install deps**:

```bash
pip install -r backend/requirements.txt
```

- **Expose to your other laptop via ngrok** (easiest demo path):

```bash
export NGROK_AUTHTOKEN=...
export USE_NGROK=1
python backend/colab_run.py
```

It will print **`Public URL: https://....ngrok-free.app`**. Use that as your frontend’s API base URL and call:
- `GET /health`
- `POST /recommend` with JSON: `{"query":"..."}`

Frontend
- Open terminal
