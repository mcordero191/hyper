import json

def get_version_history(json_file):
    """
    Read version history from a JSON file and return a history string.
    The JSON file is expected to have a key "history" that is a list of entries.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        history_lines = []
        
        for entry in data.get("history", []):
            line = f"{entry['timestamp']}: Version {entry['version']} - {entry['description']}"
            history_lines.append(line)
            
        return "\n".join(history_lines)
    
    except FileNotFoundError:
        return "No version history file found."

def get_current_version(json_file):
    """
    Return the latest version from the JSON file.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        history = data.get("history", [])
        if history:
            return history[-1]["version"]
        else:
            return "unknown"
        
    except FileNotFoundError:
        
        return "1.0.0"