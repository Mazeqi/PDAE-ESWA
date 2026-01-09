import json


class Config(object):
    """Base class for experimental setting/configuration."""

    def __init__(self):
        pass
    
    def load_config(self, import_json = ""):
        """Load settings dict from import_json (path/filename.json) JSON-file."""
        self.settings = {}
        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        # = settings
        
        for key, value in settings.items():
           self.settings[key] = value

    def save_config(self, out_path):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(out_path + '/config.json', 'w', encoding='utf8') as fp:
            json.dump(self.settings, fp, indent = 4)
