FROM julia:1.10.4

# HTTP port
EXPOSE 8000
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install git -y
# add a new user called "MLCourse"
RUN useradd -ms /bin/bash MLCourse
# set the current directory
WORKDIR /home/MLCourse
# run the rest of commands as MLCourse user
USER MLCourse
RUN git init
RUN git remote add origin https://github.com/jbrea/MLCourse
RUN git fetch
RUN git checkout -t origin/main
# copy the contents of the github repository into /home/MLCourse
# COPY --chown=MLCourse . /home/MLCourse
ENV html_export=true
ENV JULIA_PKG_DEVDIR=/home
ENV DISABLE_CACHE_SAVE=
COPY --chown=MLCourse notebooks/.cache notebooks/.cache
# COPY --chown=MLCourse flies flies

# Initialize the julia project environment that will be used to run the bind server.
# RUN julia --project=/home/MLCourse -e "import Pkg; Pkg.instantiate(); Pkg.precompile(); using MLCourse"
RUN julia -e 'import Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse")); Pkg.instantiate()'
RUN julia -e 'import Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse")); using Pluto, MLCourse, MLJ, MLJLinearModels; session = Pluto.ServerSession(); session.options.server.port = 40404; session.options.server.launch_browser = false; session.options.security.require_secret_for_access = false; t = @async Pluto.run(session); sleep(30); @async Base.throwto(t, InterruptException());'
RUN julia -e 'import Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse", "RLEnv")); Pkg.instantiate();'
RUN julia -e 'import Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse", "flies")); Pkg.instantiate();'

# The "default command" for this docker thing.
CMD ["julia", "--project=/home/MLCourse", "-e", "import PlutoSliderServer; PlutoSliderServer.run_directory(\".\"; SliderServer_port=8000, SliderServer_exclude = [\"notebooks/rnn.jl\", \"extras/transfer_learning.jl\", \"extras/generative_models.jl\"], Export_exclude = [\"notebooks/rnn.jl\", \"extras/transfer_learning.jl\", \"extras/generative_models.jl\"], SliderServer_host=\"0.0.0.0\", Export_slider_server_url=\"https://bio322.epfl.ch/\")"]
