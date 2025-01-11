def validate_spacecraft_data(data):
    """
    Validate spacecraft data
    Returns (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"
    
    if 'name' not in data:
        return False, "Name is required"
        
    if not isinstance(data['name'], str):
        return False, "Name must be a string"
        
    if len(data['name']) < 1:
        return False, "Name cannot be empty"
        
    return True, None
