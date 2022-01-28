def setup_plutoserver():
  return {
    "command": ["julia", "--optimize=0", "--sysimage=/home/jovyan/MLCourse/precompile/mlcourse.so", "--project=/home/jovyan/MLCourse", "-e", "import Pluto; Pluto.run(host=\"0.0.0.0\", port={port}, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false, dismiss_update_notification=true, show_file_system=false)"],
    "timeout": 60,
    "launcher_entry": {
        "title": "MLCourse",
    },
  }
