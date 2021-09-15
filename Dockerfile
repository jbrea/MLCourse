FROM julia:1.6.2

# HTTP port
EXPOSE 1234
RUN apt-get update -y && apt-get upgrade -y
# add a new user called "pluto"
RUN useradd -ms /bin/bash pluto
# set the current directory
WORKDIR /home/pluto
# run the rest of commands as pluto user
USER pluto
# copy the contents of the github repository into /home/pluto
COPY --chown=pluto . /home/pluto/MLCourse
ENV html_export=true
ENV JULIA_PKG_DEVDIR=/home/pluto

# Initialize the julia project environment that will be used to run the bind server.
RUN julia --project=/home/pluto/MLCourse/pluto_deployment_environment -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"

# The "default command" for this docker thing.
CMD ["julia", "--project=/home/pluto/MLCourse/pluto_deployment_environment", "-e", "import PlutoSliderServer; PlutoSliderServer.run_directory(\".\"; static_export = true, run_server = true, Export_baked_state = false, SliderServer_port=1234 , SliderServer_host=\"0.0.0.0\")"]

