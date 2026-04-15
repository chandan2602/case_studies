# Backend — FastAPI

## Setup & Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API runs at: http://localhost:8000

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /get_states | List all states |
| GET | /get_districts?state= | Districts for a state |
| GET | /get_villages?state=&district= | Villages for a district |
| POST | /generate_case_study | Generate DOCX or PDF report |
