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

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH
# ═══════════════════════════════════════════════════════════════════════════════

def test_tts_missing_fields(client):
    """TTS request with empty body should return 422."""
    response = client.post("/features/tts", json={})
    assert response.status_code == 422


def test_tts_success(client):
    """Valid TTS request should return 200 with audio/mpeg content."""
    fake_audio = b"fake-mp3-audio-bytes"
    with patch("app.routes.feature_routes.synthesize_speech", return_value=fake_audio):
        response = client.post("/features/tts", json={
            "text": "Hello",
            "language_code": "en",
        })
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


def test_tts_missing_api_key(client):
    """TTS should return 503 when the API key is not configured."""
    with patch("app.routes.feature_routes.synthesize_speech",
               side_effect=ValueError("API key not configured")):
        response = client.post("/features/tts", json={
            "text": "Hello",
            "language_code": "en",
        })
    assert response.status_code == 503


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_history_success(client):
    """Get history for a user should return 200 with a list."""
    mock_history = [
        {"original_text": "Hello", "translated_text": "Hola", "target_language": "es"}
    ]
    with patch("app.routes.feature_routes.get_user_history", return_value=mock_history):
        response = client.get("/features/history/user123")
    assert response.status_code == 200


def test_save_history_missing_fields(client):
    """Saving history with empty body should return 422."""
    response = client.post("/features/history", json={})
    assert response.status_code == 422


def test_save_history_success(client):
    """Valid history save should return 200."""
    mock_result = {"id": "hist123", "status": "saved"}
    with patch("app.routes.feature_routes.save_translation_history", return_value=mock_result):
        response = client.post("/features/history", json={
            "user_id": "user123",
            "original_text": "Hello",
            "translated_text": "Hola",
            "target_language": "es",
        })
    assert response.status_code == 200


def test_delete_history_success(client):
    """Deleting a history item should return 200."""
    mock_result = {"status": "deleted"}
    with patch("app.routes.feature_routes.delete_history_item", return_value=mock_result):
        response = client.delete("/features/history/hist123?user_id=user123")
    assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_dictionary_entries(client):
    """GET /dictionary/ should return 200 with items list."""
    mock_entries = [{"label": "A", "category": "letter", "media_type": "image", "media_url": "http://example.com/a.jpg"}]
    with patch("app.routes.dictionary_routes.list_entries", return_value=mock_entries):
        response = client.get("/dictionary/")
    assert response.status_code == 200
    assert "items" in response.json()


def test_get_dictionary_entry_not_found(client):
    """GET /dictionary/{id} with unknown ID should return 404."""
    with patch("app.routes.dictionary_routes.get_entry", return_value=None):
        response = client.get("/dictionary/nonexistent123")
    assert response.status_code == 404


def test_create_dictionary_entry_missing_fields(client):
    """Creating a dictionary entry with empty body should return 422."""
    response = client.post("/dictionary/", json={})
    assert response.status_code == 422


def test_create_dictionary_entry_invalid_category(client):
    """Creating an entry with an invalid category should return 422."""
    response = client.post("/dictionary/", json={
        "label": "Hello",
        "category": "invalid_category",
        "media_type": "image",
        "media_url": "http://example.com/hello.jpg",
    })
    assert response.status_code == 422


def test_create_dictionary_entry_success(client):
    """Valid dictionary entry creation should return 201."""
    mock_result = {"id": "dict123", "label": "Hello", "category": "word"}
    with patch("app.routes.dictionary_routes.create_entry", return_value=mock_result):
        response = client.post("/dictionary/", json={
            "label": "Hello",
            "category": "word",
            "media_type": "video",
            "media_url": "http://example.com/hello.mp4",
        })
    assert response.status_code == 201


def test_delete_dictionary_entry_not_found(client):
    """Deleting a non-existent dictionary entry should return 404."""
    with patch("app.routes.dictionary_routes.delete_entry",
               return_value={"error": "Entry not found"}):
        response = client.delete("/dictionary/nonexistent123")
    assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# USER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_user_profile_success(client):
    """GET /users/{user_id} should return 200 with user details."""
    mock_user = {
        "_id": "abc123",
        "username": "testuser",
        "email": "test@example.com",
        "profile_picture": None,
    }
    with patch("app.routes.user_routes.get_database") as mock_db:
        mock_db.return_value.__getitem__.return_value.find_one.return_value = mock_user
        response = client.get("/users/abc123")
    assert response.status_code == 200


def test_get_user_profile_invalid_id(client):
    """GET /users/{user_id} with a bad ID format should return 500."""
    response = client.get("/users/not-a-valid-id")
    assert response.status_code == 500


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN — FEEDBACK STATS & LIST
# ═══════════════════════════════════════════════════════════════════════════════

def test_admin_feedback_stats(client):
    """GET /feedbacks/admin/stats should return 200."""
    mock_stats = {"total": 10, "average_rating": 4.2}
    with patch("app.routes.feedback_routes.get_feedback_stats", return_value=mock_stats):
        response = client.get("/feedbacks/admin/stats")
    assert response.status_code == 200


def test_admin_list_feedbacks(client):
    """GET /feedbacks/admin/all should return 200 with items."""
    mock_items = [{"user_id": "u1", "rating": 5, "message": "Great!"}]
    with patch("app.routes.feedback_routes.list_feedbacks", return_value=mock_items):
        response = client.get("/feedbacks/admin/all")
    assert response.status_code == 200
    assert "items" in response.json()


def test_admin_dashboard_stats(client):
    """GET /users/admin/dashboard-stats should return 200."""
    mock_stats = {"total_users": 50, "active_users": 42}
    with patch("app.routes.user_routes.get_admin_dashboard_stats", return_value=mock_stats):
        response = client.get("/users/admin/dashboard-stats")
    assert response.status_code == 200
