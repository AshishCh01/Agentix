# Fix Authentication and Authorization Vulnerabilities

This plan addresses the critical security flaws identified in the system, specifically IDOR (Insecure Direct Object Reference) vulnerabilities in chat and search, lack of user synchronization with the database, and the presence of fake authentication endpoints.

## User Review Required

> [!WARNING]
> This plan will delete the `backend/app/routes/auth.py` file completely, as it contains fake endpoints (`/login` and `/signup`) that are not used by the Supabase-driven frontend. If anything currently depends on these mock endpoints, they will break.

## Proposed Changes

---

### Authentication Routes
Remove unused and insecure mock authentication endpoints.

#### [DELETE] [auth.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/routes/auth.py)
#### [MODIFY] [main.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/main.py)
- Remove the import `from app.routes import auth`
- Remove the router registration `app.include_router(auth.router, prefix=settings.API_V1_STR)`

---

### Database & User Synchronization
Fix the issue where users authenticated via Supabase do not exist in the local PostgreSQL `users` table, preventing proper foreign-key relationship constraints for sessions and documents.

#### [MODIFY] [crud.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/database/crud.py)
- Add a new async function `sync_user` using PostgreSQL's `INSERT ... ON CONFLICT DO UPDATE` (upsert) to safely create or update the user record based on the JWT payload.

#### [MODIFY] [dependencies.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/auth/dependencies.py)
- Update the `get_current_user` dependency to also inject `db: AsyncSession = Depends(get_db)`.
- After successfully decoding the JWT (either locally or via remote fallback), call `crud.sync_user` to ensure the user exists in the `users` table before returning the user dict.

---

### Authorization & IDOR Fixes
Secure endpoints against cross-user access (IDOR). Currently, these endpoints accept a `session_id` but do not verify that the authenticated user actually owns that session.

#### [MODIFY] [chat.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/routes/chat.py)
- In the `_setup_chat_state` helper (which protects both `/chat` and `/chat/stream`), inject a query to verify that the `session_id` is owned by `current_user["user_id"]`.
- Raise a `404 Not Found` or `403 Forbidden` exception if the session doesn't belong to the user.

#### [MODIFY] [search.py](file:///c:/Users/Ashish%20Choudhary/Desktop/Agentic%20RAG/backend/app/routes/search.py)
- In the `perform_vector_search` endpoint, fetch the `ChatSession` by ID and verify it belongs to `current_user["user_id"]` before executing the vector similarity search against the document chunks.

## Verification Plan

### Automated Tests
- Create a new test script `test_security.py` (or run a suite of local tests) to simulate:
  1. Creating a user and session A.
  2. Authenticating as user B and attempting to access session A via the `/chat` and `/search` endpoints.
  3. Verifying that a `404` or `403` error is returned, confirming IDOR prevention.
  4. Verifying that the `users` table correctly populates when a new JWT is passed to the API.
