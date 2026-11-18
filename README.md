# MySkillsGuru backend

FastAPI backend for **MySkillsGuru**, an AI study companion for self-paced learners: personalised roadmaps, homework help, AI notes, practice tests and analytics.

This is its own project. It started as a trimmed copy of the learner part of the LMS (`lms-evaluation-backend`), which is **read-only reference material**: fixes are ported by hand, and the LMS is never edited from here.

| | |
|---|---|
| Port | **5051** |
| Frontend | `../myskillguru-frontend` (port 3001) |
| Python | 3.11 |
| Data | MongoDB (database `myskillguru`), Redis, Qdrant (vector store) |
| AI | Gemini (notes, quizzes, grading), Claude (homework help, fallback) |

## Run it locally

```powershell
cd myskillguru-backend
python -m venv .venv                       # first time only
.venv\Scripts\activate
pip install -r requirements-dev.txt       # app + test dependencies
copy .env.example .env                     # first time only, then fill it in
playwright install chromium                # first time only: used to render PDFs
uvicorn app.main:app --reload --port 5051
```

Check it: <http://localhost:5051/health> should return `"status":"healthy","database":"connected"`.

**Redis:** if the Redis in `REDIS_URL` can't be reached and `ENV` is not `production`, the app logs a warning and uses an in-memory fake (state is lost on restart). In production it keeps the real client and logs an error instead.

## Settings (`.env`)

Copy `.env.example`; every setting is documented there. The ones that matter:

| Setting | Notes |
|---|---|
| `MONGODB_URI`, `DB_NAME` | Use `DB_NAME=myskillguru`. Never the LMS database. |
| `JWT_SECRET_KEY` | In production (`ENV=production`) the server refuses to start unless this is random and at least 32 characters. |
| `JWT_COOKIE_NAME` | Must differ from the LMS cookie (browsers share cookies across ports on one host). |
| `CORS_ORIGINS` | Explicit origins only; `*` is rejected. Frontend is `http://localhost:3001`. |
| `REDIS_URL`, `REDIS_KEY_PREFIX` | Every Redis key is prefixed, so apps sharing a Redis never collide. |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Own collection; never the LMS's `lms_course_material_chunks`. |
| `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `IMAGEKIT_*`, `YOUTUBE_API_KEY` | AI, file storage and learning resources. |
| `OUTBOUND_FETCH_ALLOWED_HOSTS` | Extra hosts the server may fetch from (ImageKit is always allowed). |

Never commit `.env`; it is git-ignored.

## Accounts

- Only **learner** accounts exist (role 7). Public sign-up is `POST /myskillguru/register`.
- Sign-ups are **activated immediately**. This is deliberate and temporary: the admin approval panel was removed. Before a public launch, add an approval or email-verification step (see the TEMPORARY note in `app/api/routers/auth.py`). Sign-up is rate limited, but anyone can use your AI quota until then.
- Logout and password change revoke existing sessions on the server (Redis cutoff per user).

## Project layout

```
app/
  api/routers/   auth, myskillguru_auth, profile, roadmap, mock_tests,
                 ai_tutor, self_learner_analytics, self_learner_course_material
  services/      claude, gemini, roadmap_ai, pdf_render, ai_usage, rag/ (hybrid RAG)
  core/          config, security (JWT), rate_limit, redis_client
  db/            mongodb, indexes (created on startup)
  models/ schemas/ utils/ (net.py = SSRF guard, uploads.py = bounded uploads)
scripts/         one-off data migrations
tests/
```

## Tests

```powershell
pytest                        # everything (about 7 minutes)
pytest tests/test_auth.py     # one file
```

- Tests use a **real, disposable MongoDB database** (`lms_evaluation_test`, dropped after each test) on the server named in your `.env`. It refuses to run against protected names such as `nexus`.
- Redis is faked (`fakeredis`); AI calls are mocked, so no API keys are spent.

## Security notes

- Server-side fetches go through `app/utils/net.py` (blocks internal/metadata addresses, no redirects). The PDF renderer applies the same check to every request made by the HTML it renders.
- Uploads are read through `app/utils/uploads.py` (40 MB cap, HTTP 413).
- Free-text from students is fenced before it reaches an AI prompt (`fence_user_text`).
- Startup refuses weak production secrets and wildcard CORS.

## Keeping in step with the LMS

LMS fixes are copied across by hand. To compare a file, diff it against `../lms-evaluation-backend/` after normalising the brand names. The last comparison ported: analytics timezone fix, prompt fencing, bounded uploads, streaming Claude HTML generation, and the PDF renderer guard. Intentional differences: no arq worker/queue (uses FastAPI background tasks) and no institute, exam or grading code.
