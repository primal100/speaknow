def connection_error_message(error: Exception, config: dict) -> str:
    """Explain connection failures without displaying credentials or response bodies."""
    key_name = config.get("api_key_env", "API key")
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(error, KeyError) and error.args == (key_name,):
        return (
            f"The API key environment variable {key_name} is missing or empty. "
            "Set it, then restart SpeakNow so it can read the updated environment. "
            "You can also choose another service or key variable in Settings."
        )
    if status in (401, 403, "UNAUTHENTICATED", "PERMISSION_DENIED"):
        return (
            f"Authentication was rejected (HTTP {status}). Check the key in {key_name} "
            "and confirm your account has access to the selected model. "
            "Restart SpeakNow after changing environment variables."
        )
    if status == 429:
        return "The provider's rate limit or quota was reached. Check your account quota, or wait and retry."
    if isinstance(error, TimeoutError):
        return "The service did not finish connecting in time. Check your internet connection and API settings, then retry."
    return (
        f"Could not start the session ({type(error).__name__}). "
        "Check your API key, model, base URL, internet connection, and audio devices. "
        "Correct the settings and retry."
    )
