import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

port = int(os.environ.get("PORT", 10000))
uvicorn.run("reverse_bci.ui.web:app", host="0.0.0.0", port=port)
