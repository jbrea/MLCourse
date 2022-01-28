def setup_plutoserver():
  return {
    "command": ["/bin/bash", "/home/jovyan/MLCourse/runpluto.sh", "{port}"],
    "timeout": 60,
    "launcher_entry": {
        "title": "Pluto.jl",
    },
  }
