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
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── .gitignore
└── README.md

```

## Instructions
Backend
- Open terminal(Powershell)
- Navigate to backend/
- Run the server with uvicorn app:app --host 0.0.0.0 --port 8000
- ngrok http 8000

### Hosting the backend on Google Colab (recommended for GPU)
- Ensure that you are on a GPU
- Click "Run All"
- Backend server is now live

It will print **`Public URL: https://....ngrok-free.app`**. Use that as your frontend’s API base URL and call:
- `GET /health`
- `POST /recommend` with JSON: `{"query":"..."}`

Frontend
- Open a new instance of the terminal
- Run `python -m http.server 5500`
- Use `curl http://localhost:5500` or paste `http://localhost:5500` into a browser
- Paste the backend URL into its spot


