"""
test_api.py — SignBridge Backend API Tests

Tests are grouped by feature area:
  - Root endpoint
  - User registration
  - User login
  - Translation feature
  - Feedback

Heavy ML/DB calls are mocked so tests run fast in any environment.
The 'client' fixture is provided by conftest.py.
"""

from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def test_root_returns_200(client):
    """The API root should respond with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_root_message(client):
    """The API root should confirm the service is running."""
    response = client.get("/")
    assert response.json()["message"] == "SignBridge API is running!"


# ═══════════════════════════════════════════════════════════════════════════════
# USER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_register_missing_all_fields(client):
    """Registration with an empty body should return 422 Unprocessable Entity."""
    response = client.post("/users/register", json={})
    assert response.status_code == 422


def test_register_invalid_email(client):
    """Registration with a bad email format should be rejected with 422."""
    response = client.post("/users/register", json={
        "username": "testuser",
        "email": "not-a-valid-email",
        "password": "password123",
    })
    assert response.status_code == 422


def test_register_missing_password(client):
    """Registration without a password field should return 422."""
    response = client.post("/users/register", json={
        "username": "testuser",
        "email": "test@example.com",
    })
    assert response.status_code == 422


def test_register_success(client):
    """Valid registration data should create a user and return 201."""
    mock_user = {
        "user_id": "abc123",
        "username": "testuser",
        "email": "test@example.com",
    }
    with patch("app.routes.user_routes.create_user_mongo", return_value=mock_user):
        response = client.post("/users/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
        })
    assert response.status_code == 201
    assert response.json()["message"] == "User created successfully"


def test_register_duplicate_email(client):
    """Registering with an already-used email should return 400."""
    with patch("app.routes.user_routes.create_user_mongo",
               return_value={"msg": "Email already registered"}):
        response = client.post("/users/register", json={
            "username": "anotheruser",
            "email": "existing@example.com",
            "password": "password123",
        })
    assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# USER LOGIN
# ═══════════════════════════════════════════════════════════════════════════════

def test_login_missing_fields(client):
    """Login with empty body should return 422."""
    response = client.post("/users/login", json={})
    assert response.status_code == 422


def test_login_invalid_email_format(client):
    """Login with a malformed email should be rejected with 422."""
    response = client.post("/users/login", json={
        "email": "not-an-email",
        "password": "password123",
    })
    assert response.status_code == 422


def test_login_success(client):
    """Valid credentials should return 200 with user details."""
    mock_response = {
        "msg": "Login successful",
        "user_id": "abc123",
        "username": "testuser",
        "email": "test@example.com",
        "profile_picture": None,
    }
    with patch("app.routes.user_routes.login_user_mongo", return_value=mock_response):
        response = client.post("/users/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
    assert response.status_code == 200
    assert response.json()["message"] == "Login successful"
    assert response.json()["username"] == "testuser"


def test_login_wrong_credentials(client):
    """Wrong credentials should return 401 Unauthorized."""
    with patch("app.routes.user_routes.login_user_mongo",
               return_value={"msg": "Invalid credentials"}):
        response = client.post("/users/login", json={
            "email": "test@example.com",
            "password": "wrongpassword",
        })
    assert response.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION FEATURE
# ═══════════════════════════════════════════════════════════════════════════════

def test_translate_missing_fields(client):
    """Translation request with no body should return 422."""
    response = client.post("/features/translate", json={})
    assert response.status_code == 422


def test_translate_missing_target_lang(client):
    """Translation request without target_lang should return 422."""
    response = client.post("/features/translate", json={"text": "hello"})
    assert response.status_code == 422


def test_translate_success(client):
    """Valid translation request should return 200 with translated text."""
    mock_result = {"translated_text": "Hola", "source_lang": "en", "target_lang": "es"}
    with patch("app.routes.feature_routes.translate_text", return_value=mock_result):
        response = client.post("/features/translate", json={
            "text": "hello",
            "target_lang": "es",
            "source_lang": "en",
        })
    assert response.status_code == 200
    assert "translated_text" in response.json()
    assert response.json()["translated_text"] == "Hola"


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

def test_feedback_missing_fields(client):
    """Submitting feedback with empty body should return 422."""
    response = client.post("/feedbacks/", json={})
    assert response.status_code == 422


def test_feedback_success(client):
    """Valid feedback submission should return 201."""
    mock_result = {"id": "fb123", "status": "Feedback saved successfully"}
    with patch("app.routes.feedback_routes.create_feedback", return_value=mock_result):
        response = client.post("/feedbacks/", json={
            "user_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "rating": 5,
            "message": "Great sign language app!",
        })
    assert response.status_code == 201
