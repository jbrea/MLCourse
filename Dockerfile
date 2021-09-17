FROM julia:1.6.2

# HTTP port
EXPOSE 8000
RUN apt-get update -y && apt-get upgrade -y
# add a new user called "MLCourse"
RUN useradd -ms /bin/bash MLCourse
# set the current directory
WORKDIR /home/MLCourse
# run the rest of commands as MLCourse user
USER MLCourse
# copy the contents of the github repository into /home/MLCourse
COPY --chown=MLCourse . /home/MLCourse
ENV html_export=true
ENV JULIA_PKG_DEVDIR=/home

# Initialize the julia project environment that will be used to run the bind server.
RUN julia --project=/home/MLCourse -e "import Pkg; Pkg.instantiate(); Pkg.precompile(); using MLCourse"

# The "default command" for this docker thing.
CMD ["julia", "--project=/home/MLCourse", "-e", "import PlutoSliderServer; PlutoSliderServer.run_directory(\".\"; static_export = true, run_server = true, Export_baked_state = false, SliderServer_port=8000, SliderServer_host=\"0.0.0.0\", Export_slider_server_url=\"https://bio322.epfl.ch/\")"]
