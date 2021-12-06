### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b97724e4-d7b0-4085-b88e-eb3c5bcbe441
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using StatsBase, DataFrames, MLCourse, Plots
end

# ╔═╡ b3793299-c916-40d3-bd87-31153fc3781a
using PlutoUI, PlutoUI.BuiltinsNotebook.HypertextLiteral; PlutoUI.TableOfContents()

# ╔═╡ 1a17e9b2-a439-4521-842c-96ebe0378919
using ReinforcementLearning

# ╔═╡ d88c1d9b-3396-42a2-8ebd-81851f778602
begin
    local result = begin
        struct CounterButtons
            labels::Vector{String}
        end
    end

	function Base.show(io::IO, m::MIME"text/html", button::CounterButtons)
		buttons = [@htl("""<input type="button" value="$label">""")
                   for label in button.labels]
		show(io, m, @htl("""
		<span>
        $buttons
		<script>
		let count = 0
		const span = currentScript.parentElement
		span.value = [count, 0]
		Array.prototype.forEach.call(span.children, function(child, index){
			child.addEventListener("click", (e) => {
				count += 1
				span.value = [count, index + 1]
				span.dispatchEvent(new CustomEvent("input"))
				e.stopPropagation()
			})
		});
		</script>
		</span>
		"""))
	end

	Base.get(button::CounterButtons) = button.labels
	PlutoUI.BuiltinsNotebook.Bonds.initial_value(::CounterButtons) = [0, 0]
	PlutoUI.BuiltinsNotebook.Bonds.possible_values(::CounterButtons) = PlutoUI.BuiltinsNotebook.Bonds.InfinitePossibilities()
	function PlutoUI.BuiltinsNotebook.Bonds.validate_value(::CounterButtons, val)
		val isa Vector{Int} && val[1] >= 0 && val[2] >= 0
	end

	result
end

# ╔═╡ ce405f97-6d60-4ae4-b183-79e6c88d9811
md"# Chasse au trésor"

# ╔═╡ 761d690d-5c73-40dd-b38c-5af67ee837c0
begin
    function onehot(i)
        x = zeros(7)
        x[i] = 1
        x
    end
    act(state, action) = wsample(1:7, T[action, state])
    _reward(r::Number) = r
    _reward(r::AbstractArray) = rand(r)
    reward(state, action) = _reward(R[state, action])
    T = reshape([[0, .7, .3, 0, 0, 0, 0], [0, 0, 0, .3, .7, 0, 0],
                  onehot(6), onehot(6),
                  onehot(7), onehot(6),
                  onehot(6), onehot(6),
                  onehot(6), onehot(7),
                  onehot(6), onehot(6),
                  onehot(7), onehot(7)], 2, :)
    R = [-.5 -.3
         [1:4] 1
         -5  1
         1 [3:6]
         1 -5
         0 0
         0 0]
    function showQ(Q; states = ["red room", "blue without guard", "blue with guard", "green without guard", "green with guard", "treasure room", "K.O."], actions = ["left", "right"])
        df = DataFrame(Q, states)
        df.action = actions
        df
    end
    Base.@kwdef mutable struct ChasseAuTresorEnv
        state::Int = 1
        reward::Float64 = 0
        episode_recorder = Tuple{Int, Int, Float64, Int}[]
        action::Union{Int,Nothing} = nothing
    end
    function act!(env::ChasseAuTresorEnv, a)
        length(env.episode_recorder) == 2 && return
        s = env.state
        r = reward(s, a)
        env.reward = r
        s′ = act(env.state, a)
        env.state = s′
        push!(env.episode_recorder, (s, a, r, s′))
    end
    reward(env::ChasseAuTresorEnv) = env.reward
    state(env::ChasseAuTresorEnv) = env.state
    function reset!(env::ChasseAuTresorEnv)
        env.state = 1
        env.reward = 0
        empty!(env.episode_recorder)
    end
    function distributed_representation(s)
        [s == 1; # 1 if in room 1
         s ∈ (2, 3); # 1 if in room 2
         s ∈ (4, 5); # 1 if in room 3
         s == 6; # 1 if in treasure room
         s == 7; # 1 if KO
         s ∈ (3, 5)] # 1 if guard present
    end
    chasse = ChasseAuTresorEnv()
end;

# ╔═╡ 7cf26d6c-5a67-4bd6-8ef2-34559b53685b
@bind chasse_actions CounterButtons(["open left door", "open right door",
                                     "reset episode", "reset learner"])

# ╔═╡ b0b89b30-38f1-45c1-8c7a-796ea2f41e8d
md"## Learning Q-Values with Monte Carlo Estimation
In Monte Carlo estimation, the Q-values ``Q(s, a)`` of state-action pair ``(s, a)`` are updated every time action ``a`` is taken in state ``s`` such that the current estimate is given by the average of all cumulative returns obtained so far when taking action ``a`` in state ``s``.

Additionally to an implementation of Monte Carlo Estimation of Q-values you see in the following cell a way to organize code in Julia using custom types.
We define the structure `MCLearner` to have the fields `N` and `Q` (with some default values adapted to our chasse au trésor) and some functions to update objects of type `MCLearner`. We also write doc-strings for our new functions. Open the live docs and click on the `update!` function to see the rendered doc-string.
"

# ╔═╡ 7c756af7-08a0-4878-8f53-9eaa79c16c1a
begin
    Base.@kwdef struct MCLearner # this defines a type MCLearner
        ns = 7                 # default initial number of states
        na = 2                 # default initial number of actions
        N = zeros(Int, na, ns) # default initial counts
        Q = zeros(na, ns)      # default initial Q-values
    end
	"""
	    update!(l::MCLearner, a, s, r)

	Updates an `MLClearner` object `l` for a given
	action `a`, state `s` and reward `r`.
	"""
    function update!(learner::MCLearner, a, s, r)
        N, Q = learner.N, learner.Q
        n = N[a, s]
        Q[a, s] = 1/(n+1) * (n * Q[a, s] + r) # efficient averaging
        N[a, s] += 1
        learner
    end
	"""
	    update!(l::MCLearner, episode)

	Updates an `MCLearner` oject `l` for a full episode.
	"""
    function update!(learner::MCLearner, episode)
		G = 0 # initialise the cumulative reward
        for (s, a, r, ) in reverse(episode)
			G += r
            update!(learner, a, s, G)
        end
        learner
    end
    reset!(learner::MCLearner) = learner.Q .= learner.N .= 0
    mclearner = MCLearner() # create an object of type MCLearner
end;

# ╔═╡ 6dced47e-cc0f-4ae5-bdd0-a551c6d5a5b3
md"## An Epsilon-Greedy Agent

Up until now we took the actions. Here we introduce a very simple agent that takes action currently believed to be the best one with probability ``1 - \epsilon`` and a random action with probability ``\epsilon``.
"

# ╔═╡ 64863a37-1c25-4d1c-9f2d-33d87b4039a3
function epsilon_greedy_policy(q; ϵ = .25)
    if rand() < ϵ # this happens with probability ϵ
        rand(1:length(q))  # random exploration
    else
        rand(findall(==(maximum(q)), q)) # greedy exploitation with ties broken
    end
end

# ╔═╡ 0537554c-76e4-4827-b92a-bf5b944685c7
md"Now we let this agent play the game for 10^5 episodes and learn the Q-values with Monte Carlo estimation."

# ╔═╡ dc7c5cb0-30b2-4427-8aeb-f312a88effd1
md"# Q-Learning

Q-learning is an alternative method for updating Q-values that relies on the idea that ``Q(s_t, a_t)`` for the optimal policy can be estimated by knowing ``r_t`` and ``\max_a Q(s_{t+1}, a``.
"

# ╔═╡ e39227c8-8148-43cb-9351-774682b65646
begin
    Base.@kwdef struct QLearner # this defines a type QLearner
        ns = 7                 # default initial number of states
        na = 2                 # default initial number of actions
        Q = zeros(na, ns)      # default initial Q-values
        λ = 0.01
    end
	"""
	    update!(l::QLearner, a, s, r)

	Updates an `MLClearner` object `l` for a given
	state `s`, action `a`, reward `r` and next state `s′`.
	"""
    function update!(learner::QLearner, s, a, r, s′)
        Q, λ = learner.Q, learner.λ
        δ = (r + maximum(Q[:, s′]) - Q[a, s])
        Q[a, s] = Q[a, s] + λ * δ
        learner
    end
    reset!(learner::QLearner) = learner.Q .= 0
    qlearner = QLearner()
end;

# ╔═╡ bf600a9b-484e-4a7b-bf3b-409ebad51cd0
md"""
If you want to see Q-learning in action, choose "qlearner" below and play our
chasse au trésor. The displayed Q-values will then be for the `qlearner`.

Learner $(@bind learner Select(["mclearner", "qlearner"]))

Below you see the Q-values of an epsilon-greedy policy played for ``10^6`` steps.
Note that is comes close to the optimal solution. This is different than what we found with the Monte Carlo Learner. The reason is that the Monte Carlo Learner estimates the Q-values for the epsilon-greedy policy, whereas the Q-Learner estimates the Q-values for the greedy policy, even though the actions are choosen according to the epsilon-greedy policy. The Monte Carlo Learner is called *on-policy* because it evaluates the policy used for action selection. The Q-Learner is called *off-policy*, because it evaluates the greedy policy, while selecting actions with the epsilon-greedy policy.
"""

# ╔═╡ 5a705ba7-dd9c-411c-a910-659fb1ec9f82
md"Below you see the Q-values of the \`$learner\`."

# ╔═╡ e876c526-30f9-458d-abf5-e20e6aa0268e
let
    chasse_actions
    if learner == "mclearner"
        showQ(mclearner.Q)
    else
        showQ(qlearner.Q)
    end
end

# ╔═╡ aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
md"# Learning to Play Tic-Tac-Toe

In the following example we will learn to play Tic-Tac-Toe with a Monte Carlo Learner.
The Tic-Tac-Toe environment is loaded from the [`ReinforcementLearning` package](https://juliareinforcementlearning.org/). This package also provides multiple helper functions to interact with this environment, like `state`, `is_terminated`, `legal_action_space`, `ReinforcementLearning.reward`, `ReinforcementLearning.reset!` etc.
"

# ╔═╡ ae72b2c3-fe7b-4ffb-b45c-e589529209c7
tictactoe = TicTacToeEnv();

# ╔═╡ d094b8d7-5cc7-4bc0-a406-2e3c85f9fb60
md"To perform an action we can simply call the environment with an integer from 1 to 9.
1 means placing a cross `x` or nought `o` (depending on whose turn it is) at the top left corner, 3 is at the bottom left corner, 7 top right and 9 bottom right."

# ╔═╡ 4b90f08e-0183-4f8b-a6cd-e66b6938f4c8
legal_action_space(tictactoe) # now action 5 is no longer available

# ╔═╡ 59029032-1c91-4da1-a61b-6a56449dcd2c
md"Let us now actually play the game. You can choose different modes below to play against yourself (or another human player), against the computer (trained in self-play with our `MCLearner`; see below) or computer against computer. To advance the game when the computer plays against itself you have to use the `step` button."

# ╔═╡ 94c0859f-f2d4-4b00-9793-14c823bbc705
@bind tictactoe_action CounterButtons([string.(1:9); "reset game"; "step"])

# ╔═╡ 4a9fb8a0-81fb-4e59-8208-61df1dbd8255
md"""Player Cross: $(@bind player1 Select(["human", "machine"])) Player Nought: $(@bind player2 Select(["human", "machine"]))"""

# ╔═╡ d99a218a-56e8-4081-bd1a-ba7729f529cf
md"The computer player is using a greedy policy with Q-values learned in self-play with the code below."

# ╔═╡ e61c6d43-a097-4f44-a343-05371b4932ef
begin
    is_terminated(env::TicTacToeEnv) = RLBase.is_terminated(env)
    act!(env::TicTacToeEnv, a) = env(a)
    reset!(env::TicTacToeEnv) = RLBase.reset!(env)
    reward(env::TicTacToeEnv) = RLBase.reward(env)
    state(env::TicTacToeEnv) = RLBase.state(env, Observation{Int}())
end;

# ╔═╡ b6b835d1-8f84-4148-8d5b-c7aea6b0c312
md"""# Exercises
## Conceptual

Consider an agent that experiences episode 1:

``((S_1 = s_1, A_1 = a_1, R_1 = 2), (S_2 = s_7, A_2 = a_2, R_2 = 1), (S_3 = s_6, A_2 = a_2, R_3 = 4), (S_4 = s_3, A_4 = a_1, R_4 = 0))``

and episode 2:

``((S_1 = s_1, A_1 = a_1, R_1 = 1)), (S_2 = s_6, A_2 = a_1, R_2 = -1), (S_3 = s_2, A_3 = a_2, R_3 = 1))``.

(a) Compute ``Q(s_1, a_1)`` with Monte Carlo Estimation.

(b) Compute ``Q(s_1, a_1)`` with Q-Learning, learning rate ``\lambda = 0.5`` and initial ``Q(s_1, a_1) = 0``.

## Applied

Consider the cliff-walking task shown below. This is a standard episodic task with an agent (red point) moving around in a grid world with start (bottom left) and goal state (green square at bottom right), and the usual actions causing movement up, down, right, and left. Reward is -1 on all transitions except those into the cliff region marked black and the goal state at the bottom right. Stepping into this region incurs a reward of -100. Episodes end when the agent steps into the cliff or reaches the goal at the bottom right; in this case `is_terminated(environment) == true` and `reset!(environment)` sends the agent back to the start state at the bottom left.

(a) Determine the cumulative reward that would be obtained when following the optimal strategy in this task.

(b) Train a Monte Carlo learner with epsilon-greedy exploration (``\epsilon = 0.1``) for ``10^3`` episodes. Keep track of the cumulative reward in each episode. *Hints:* You can use the `environment = CliffWalkingEnv()` and the usual functions like `state(environment)`, `reward(environment)` and `act!(environment, action)` etc. Because we do not know the length of the episodes, you can use the following pattern:
```julia
while true
    ...
    if is_terminated(environment)
        ...
        break
    end
end
```

(c) Plot the cumulative reward per episode, i.e. x-axis: episode, y-axis: cumulative reward.

(d) Use the following functions to plot the learned greedy policy and the Q values of the greedy policy (i.e. the maximal Q values for each state)
```julia
function plot_greedy_policy(environment, Q)
	[x[1] == 1 ? "<" : x[1] == 2 ? ">" : x[1] == 3 ? "∧" : "∨"
		for x in argmax(Q, dims = 1)][environment.params.states] |>
	string |>
	s -> replace(s, r"[\\"\[\]]" => "") |>
	s -> replace(s, "; " => "\n") |>
	Text
end
function plot_q(environment, Q)
    heatmap(maximum(Q, dims = 1)[environment.params.states], yflip = true)
end
```

(e) Explain why the Monte Carlo learner failed to learn a good policy.

(f) With `reset!(environment)` the agent is put to the start state at the end of each episode. You can use instead the following function to reset the agent to a random position.
```julia
function random_reset!(env::CliffWalkingEnv)
	env.position = rand(CartesianIndices((4, 12)))
end
```
Run ``10^5`` episodes with this random resetting of the environment and plot again the learned greedy policy and Q values.

(g) Repeat (b-d) for a Q learner.

(h) Explain, why the cumulative reward of the Q learner is typically lower than the cumulative reward of the optimal policy, even though the Q learner finds the optimal greedy policy.

(i) Explain, why the Monte Carlo learner with random reset learns a greedy policy that does not follow as closely the cliff as the greedy policy learned with Q learning.
"""


# 2. Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as different from each other as possible. The framework is abstract and flexible and can be applied in many different ways. Stretch its limits in some way in at least one of your examples.
# 3. Show for infinite-horizon problems with discount factor ``0 < \\gamma < 1`` the Bellman equations for a determinstic policy ``\\pi`` are given by ``Q_\\pi(s, a) = \\bar r(s, a) + \\gamma \\sum_{s'}P(s'|s, a)Q_\\pi(s', \\pi(s'))``. Start with the definition ``Q_\\pi(S_t, A_t) = \\mathrm{E}\\left[R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\gamma^3R_{t+4} + \\cdots \\right]`` and show all the intermediate steps that lead to the result.

# ╔═╡ 8c0a5e46-c790-4cec-ac57-1b2813b81358
@bind cw_action CounterButtons(["left", "right", "up", "down", "reset"])

# ╔═╡ 7c4a9aff-c3c1-48ef-8e83-9d5aa7e75b03
begin
    Base.@kwdef struct CliffWalkingParams
        nx::Int = 4
        ny::Int = 12
        states::LinearIndices{2, NTuple{2, Base.OneTo{Int}}} = LinearIndices((4, ny))
        start::CartesianIndex{2} = CartesianIndex(4, 1)
        goal::CartesianIndex{2} = CartesianIndex(4, ny)
        actions::Vector{CartesianIndex{2}} = [CartesianIndex(0, -1),  # left
                                              CartesianIndex(0, 1),   # right
                                              CartesianIndex(-1, 0),  # up
                                              CartesianIndex(1, 0),   # down
                                             ]
    end
    Base.@kwdef mutable struct CliffWalkingEnv <: AbstractEnv
        params::CliffWalkingParams = CliffWalkingParams()
		position::CartesianIndex{2} = params.start
        cumulative_reward::Int = 0
	end
    iscliff(p, nx, ny) = p[1] == nx && p[2] > 1 && p[2] < ny
    iscliff(env::CliffWalkingEnv) = iscliff(env.position, env.params.nx, env.params.ny)
	function act!(env::CliffWalkingEnv, a::Int)
        x, y = Tuple(env.position + env.params.actions[a])
		env.position = CartesianIndex(min(max(x, 1), env.params.nx),
                                      min(max(y, 1), env.params.ny))
	end
    act!(::CliffWalkingEnv, ::Nothing) = nothing
    state(env::CliffWalkingEnv) = env.params.states[env.position]
    reward(env::CliffWalkingEnv) = env.position == env.params.goal ? 0.0 : (iscliff(env) ? -100.0 : -1.0)
    is_terminated(env::CliffWalkingEnv) = env.position == env.params.goal || iscliff(env)
    reset!(env::CliffWalkingEnv) = env.position = env.params.start
    function Plots.plot(env::CliffWalkingEnv)
        m = float.((!iscliff).(CartesianIndices((env.params.nx, env.params.ny)),
                               env.params.nx, env.params.ny))
        m[env.params.nx, env.params.ny] = 2
        heatmap(m;
                yflip = true, legend = false, aspect_ratio = 1,
                c = palette([:black, :yellow, :green]),
            axis = false, ticks = false)
        for i in 0:env.params.nx
            plot!([0.5, env.params.ny + .5], [i + .5, i + .5], c = :black)
        end
        for i in 0:env.params.ny
            plot!([i + .5, i + .5], [0.5, env.params.nx + .5], c = :black)
        end
        scatter!([env.position[2]], [env.position[1]],
                 markersize = 10, c = :red)
    end
    cwenv = CliffWalkingEnv();
end;

# ╔═╡ e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
let
    chasse.action = chasse_actions[1] == 0 ? 3 : chasse_actions[2]
    if chasse.state ≤ 5 || (isa(chasse.action, Int) && chasse.action > 2)
        _learner = if learner == "mclearner"
            if length(chasse.episode_recorder) > 0 && chasse.action == 3
                update!(mclearner, chasse.episode_recorder)
            end
            mclearner
        else
            if length(chasse.episode_recorder) > 0
                update!(qlearner, last(chasse.episode_recorder)...)
            end
            qlearner
        end
        if chasse.action == 3
            reset!(chasse)
        elseif chasse.action == 4
            reset!(_learner)
        elseif isa(chasse.action, Int)
            act!(chasse, chasse.action)
        end
    end
    d = distributed_representation(chasse.state)
    room = findfirst(d)
    guard = d[6]
    guard_door = room > 2
    gold = d[4]
    room_colors = [:red, :blue, :green, :orange, :black]
    plot(xlim = (0, 1), ylim = (0, 1), size = (600, 400),
         bg = room_colors[room], framestyle = :none, legend = false)
    if room < 4
        plot!([.1, .1, .4, .4], [0, .7, .7, 0], w = 5, c = :black)
        plot!([.14, .12, .17], [.37, .35, .35], w = 4, c = :black)
        plot!(.5 .+ [.1, .1, .4, .4], [0, .7, .7, 0], w = 5, c = :black)
        plot!(.5 .+ [.14, .12, .17], [.37, .35, .35], w = 4, c = :black)
    end
    if guard
        xshift = guard_door * .5
        plot!(xshift .+ [.14, .17, .25, .33, .37], [0, 0, .3, 0, 0], w = 5, c = :black)
        plot!(xshift .+ [.25, .25], [.3, .52], w = 5, c = :black)
        scatter!(xshift .+ [.25], [.58], markersize = 25, markerstrokewidth = 0, c = :black)
        plot!(xshift .+ [.33, .33, .18, .18], [.23, .47, .47, .23],
              w = 5, c = :black)
    end
    if gold
        x = [.25, .15, .2, .3, .35, .25] .+ .25
        y = [.3, .1, .2, .2, .1, .1]
        r = floor(Int, chasse.reward)
        x = x[1:r]
        y = y[1:r]
        scatter!(x, y, markerstrokewidth = 3, c = :yellow, markersize = 28)
    end
    if room == 5
        scatter!([0], [0], c = :black, legend = false, markerstrokewidth = 0) # dummy
        annotate!([(.5, .5, "K.O.", :red)])
    end
    rs = length(chasse.episode_recorder) == 0 ? [0] : getindex.(chasse.episode_recorder, 3)
    annotate!([(.5, .9, "reward = $(chasse.reward)", :white),
               (.5, .8, "cumulative reward = $(join(rs, " + ")) = $(sum(rs))", :white)
              ])
end

# ╔═╡ 712c2a9e-4413-4d7a-b729-cfb219723256
let mclearner = MCLearner(na = 2, ns = 7),
    chasse = ChasseAuTresorEnv()
    for _ in 1:10^5
		reset!(chasse)
        episode = []
        for steps in 1:2
            s = state(chasse)
            a = epsilon_greedy_policy(mclearner.Q[:, s])
            act!(chasse, a)
            r = reward(chasse)
            push!(episode, (s, a, r))
        end
        update!(mclearner, episode)
    end
    showQ(mclearner.Q)
end

# ╔═╡ bd8557bc-86f2-4ccc-93e9-a6bd843e80be
let qlearner = QLearner(na = 2, ns = 7),
    chasse = ChasseAuTresorEnv()
    for _ in 1:10^6
        reset!(chasse)
        for steps in 1:2
            s = state(chasse)
            a = epsilon_greedy_policy(qlearner.Q[:, s])
            act!(chasse, a)
            r = reward(chasse)
            s′ = state(chasse)
            update!(qlearner, s, a, r, s′)
        end
    end
    showQ(qlearner.Q)
end

# ╔═╡ ddd02bd9-9577-44a8-8bb6-2b1d11938121
state(tictactoe)

# ╔═╡ aca72848-c488-46d1-a3df-0cf6fb04f4a3
let
	act!(tictactoe, 5) # place a cross in the center
	Text(RLBase.state(tictactoe)) # diplay the state nicely
end

# ╔═╡ 86a78734-2b31-4ee5-8560-8a9388672b45
reset!(tictactoe);

# ╔═╡ 6a96c33a-b6b3-4a0a-83c8-a0df113887d0
mcl = let
    mcl = MCLearner(na = 9, ns = 5478) # total number of actions and states
    episode_cross = [] # buffer where we store the positions and actions of cross
    episode_nought = [] # buffer where we store the positions and actions of nought
    for game in 1:10^5
        reset!(tictactoe)
        move = 0
        while true
            move += 1
            legal_a = legal_action_space(tictactoe)
            s = state(tictactoe)
            i = epsilon_greedy_policy(mcl.Q[legal_a, s])
            a = legal_a[i]
            act!(tictactoe, a)
            if is_terminated(tictactoe)
                r = reward(tictactoe) # reward of next player
                if isodd(move)
                    r *= -1 # cross finished
                end
                push!(episode_cross, (s, a, r))
                push!(episode_nought, (s, a, -r))
                break
            else
                push!(isodd(move) ? episode_cross : episode_nought,
                      (s, a, 0.))
            end
        end
        update!(mcl, episode_cross)
        update!(mcl, episode_nought)
        empty!(episode_cross)
        empty!(episode_nought)
    end
    mcl
end;

# ╔═╡ c692cc6e-dbb5-40e9-aeaa-486b098c3af1
begin
    import ReinforcementLearning.ReinforcementLearningEnvironments: Cross, Nought
    function autoplaying(player1, player2, tictactoe)
        is_terminated(tictactoe) && return false
        player1 == "machine" && player2 == "machine" && return true
        player1 == "machine" && current_player(tictactoe) == Cross() && return true
        player2 == "machine" && current_player(tictactoe) == Nought() && return true
        false
    end
end;


# ╔═╡ ebebd97a-9dc2-4b39-a998-9279d52c57e5
let
    a = last(tictactoe_action)
    if a in legal_action_space(tictactoe)
        act!(tictactoe, a)
    elseif a == 10
        reset!(tictactoe)
    end
    if autoplaying(player1, player2, tictactoe)
        legal_a = legal_action_space(tictactoe)
        a2 = legal_a[argmax(mcl.Q[legal_a, state(tictactoe)])]
        act!(tictactoe, a2)
    end
    cpl = split(string(current_player(tictactoe)), ".")[end][1:end-2]
    s = if is_terminated(tictactoe)
        if ReinforcementLearning.reward(tictactoe) == -1
            "$cpl lost the game."
        else
            "The game ended in a draw."
        end
    else
        "\nCurrent player: $cpl"
    end
    legal_a = legal_action_space(tictactoe)
    s *= "\n\n" * RLBase.state(tictactoe)
    s *= "\n\n Legal actions:\n\n$(join([join([k in legal_a ? string(k) : " " for k in 3 .* (0:2) .+ j], " ") for j in 1:3], "\n"))"
    s *= "\n\n Game state: $(state(tictactoe))"
    Text(s)
end

# ╔═╡ 61b3c6ca-c680-41d1-9eb5-6ec2f799f0d1
let
    a = last(cw_action)
    if a == 5
        r = 0
        cwenv.cumulative_reward = 0
        reset!(cwenv)
    elseif a == 0
        r = 0
    elseif !is_terminated(cwenv)
        act!(cwenv, a)
        r::Int = reward(cwenv)
        cwenv.cumulative_reward += r
    end;
    plot(cwenv)
    plot!(size = (700, 400))
    annotate!([(cwenv.params.ny ÷ 2, -1, "reward = $r"),
               (cwenv.params.ny ÷ 2, -.2, "cumulative reward = $(cwenv.cumulative_reward)")])
end

# ╔═╡ 8bd459cb-20bb-483e-a849-e18caae3beef
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 412d8fcb-8f98-43b6-9235-a4c228317427
MLCourse.footer()

# ╔═╡ Cell order:
# ╠═b97724e4-d7b0-4085-b88e-eb3c5bcbe441
# ╟─b3793299-c916-40d3-bd87-31153fc3781a
# ╟─d88c1d9b-3396-42a2-8ebd-81851f778602
# ╟─ce405f97-6d60-4ae4-b183-79e6c88d9811
# ╟─761d690d-5c73-40dd-b38c-5af67ee837c0
# ╟─e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
# ╟─7cf26d6c-5a67-4bd6-8ef2-34559b53685b
# ╟─5a705ba7-dd9c-411c-a910-659fb1ec9f82
# ╟─e876c526-30f9-458d-abf5-e20e6aa0268e
# ╟─b0b89b30-38f1-45c1-8c7a-796ea2f41e8d
# ╠═7c756af7-08a0-4878-8f53-9eaa79c16c1a
# ╟─6dced47e-cc0f-4ae5-bdd0-a551c6d5a5b3
# ╠═64863a37-1c25-4d1c-9f2d-33d87b4039a3
# ╟─0537554c-76e4-4827-b92a-bf5b944685c7
# ╠═712c2a9e-4413-4d7a-b729-cfb219723256
# ╟─dc7c5cb0-30b2-4427-8aeb-f312a88effd1
# ╠═e39227c8-8148-43cb-9351-774682b65646
# ╟─bf600a9b-484e-4a7b-bf3b-409ebad51cd0
# ╠═bd8557bc-86f2-4ccc-93e9-a6bd843e80be
# ╟─aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
# ╠═1a17e9b2-a439-4521-842c-96ebe0378919
# ╠═ae72b2c3-fe7b-4ffb-b45c-e589529209c7
# ╠═ddd02bd9-9577-44a8-8bb6-2b1d11938121
# ╟─d094b8d7-5cc7-4bc0-a406-2e3c85f9fb60
# ╠═aca72848-c488-46d1-a3df-0cf6fb04f4a3
# ╠═4b90f08e-0183-4f8b-a6cd-e66b6938f4c8
# ╠═86a78734-2b31-4ee5-8560-8a9388672b45
# ╟─59029032-1c91-4da1-a61b-6a56449dcd2c
# ╟─94c0859f-f2d4-4b00-9793-14c823bbc705
# ╟─4a9fb8a0-81fb-4e59-8208-61df1dbd8255
# ╟─ebebd97a-9dc2-4b39-a998-9279d52c57e5
# ╟─d99a218a-56e8-4081-bd1a-ba7729f529cf
# ╠═6a96c33a-b6b3-4a0a-83c8-a0df113887d0
# ╟─e61c6d43-a097-4f44-a343-05371b4932ef
# ╟─c692cc6e-dbb5-40e9-aeaa-486b098c3af1
# ╟─b6b835d1-8f84-4148-8d5b-c7aea6b0c312
# ╟─8c0a5e46-c790-4cec-ac57-1b2813b81358
# ╟─61b3c6ca-c680-41d1-9eb5-6ec2f799f0d1
# ╟─7c4a9aff-c3c1-48ef-8e83-9d5aa7e75b03
# ╟─8bd459cb-20bb-483e-a849-e18caae3beef
# ╟─412d8fcb-8f98-43b6-9235-a4c228317427
