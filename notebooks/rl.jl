### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ b97724e4-d7b0-4085-b88e-eb3c5bcbe441
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using PlutoUI, StatsBase, DataFrames
    PlutoUI.TableOfContents()
end

# ╔═╡ 26918860-b3c3-454a-8767-2116b2ece629
using Flux

# ╔═╡ 1a17e9b2-a439-4521-842c-96ebe0378919
using ReinforcementLearning

# ╔═╡ 8bd459cb-20bb-483e-a849-e18caae3beef
begin
    using MLCourse, Plots
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ ce405f97-6d60-4ae4-b183-79e6c88d9811
md"# Chase au trésor"

# ╔═╡ 12697628-d520-4c3a-9e33-f7c8e17f0dea
@bind action Radio(["open left door", "open right door", "reset episode", "reset learner"],
                   default = "reset episode")

# ╔═╡ 86a4f1ef-c81c-4888-95ed-15269d38fbea
@bind do_it Button("go")

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
end;

# ╔═╡ 2b7329a7-177a-468d-b9b9-1cba648ba570
begin
    Base.@kwdef mutable struct GameState
        state::Int = 1
        reward::Float64 = 0
        episode_recorder = Tuple{Int, Int, Float64, Int}[]
        action::String = ""
    end
    function act!(game_state::GameState, a)
        length(game_state.episode_recorder) == 2 && return
        s = game_state.state
        r = reward(s, a)
        game_state.reward = r
        s′ = act(game_state.state, a)
        game_state.state = s′
        push!(game_state.episode_recorder, (s, a, r, s′))
    end
    function reset!(game_state::GameState)
        game_state.state = 1
        game_state.reward = 0
        empty!(game_state.episode_recorder)
    end
    game_state = GameState()
end;

# ╔═╡ c50abfb9-6f0a-4a80-a54d-cd04e7f99aa3
game_state.action = action;

# ╔═╡ b0b89b30-38f1-45c1-8c7a-796ea2f41e8d
md"## Learning Q-Values with Monte Carlo Estimation

Additionally to an implementation of Monte Carlo Estimation of Q-values you see in the following cell a way to organize code in Julia using custom type
We define the structure `MCLearner` to have the fields `N` and `Q` (with some default values adapted to our chase au trésor) and some functions to update objects of type `MCLearner`. We also write doc-strings for our new functions. Open the live docs and point the mouse pointer on the `update!` function to see the rendered doc-string.
"

# ╔═╡ 7c756af7-08a0-4878-8f53-9eaa79c16c1a
begin
    Base.@kwdef struct MCLearner # this defines a type MCLearner
        N = zeros(Int, size(T)...) # default initial counts
        Q = zeros(size(T)...)      # default initial Q-values
    end
	"""
	    update!(l::MCLearner, a, s, r)

	Updates an `MLClearner` object `l` for a given
	action `a`, state `s` and reward `r`.
	"""
    function update!(l::MCLearner, a, s, r)
        N, Q = l.N, l.Q
        n = N[a, s]
        Q[a, s] = 1/(n+1) * (n * Q[a, s] + r)
        N[a, s] += 1
        l
    end
	"""
	    update!(l::MCLearner, episode)

	Updates an `MCLearner` oject `l` for a full episode.
	"""
    function update!(l::MCLearner, episode)
		length(episode) < 2 && return l # don't update if the episode has not ended
		G = 0 # initialise the cumulative reward
        for (s, a, r, _) in reverse(episode)
			G += r
            update!(l, a, s, G)
        end
        l
    end
    reset!(l::MCLearner) = l.Q .= l.N .= 0
    mclearner = MCLearner() # create an object of type MCLearner
end;

# ╔═╡ fe5ae6b8-bc5e-453a-96fd-f3e7dab135e5
begin
    function showQ(Q)
        df = DataFrame([Q; (Q[1, :] .- Q[2, :])'], ["red room", "blue without guard", "blue with guard", "green without guard", "green with guard", "treasure room", "K.O."])
        df.action = ["left", "right", "left - right"]
        df
    end
end;

# ╔═╡ 6dced47e-cc0f-4ae5-bdd0-a551c6d5a5b3
md"## An Epsilon-Greedy Agent"

# ╔═╡ 64863a37-1c25-4d1c-9f2d-33d87b4039a3
function epsilon_greedy_policy(q; ϵ = .25)
    if rand() < ϵ # this happens with probability ϵ
        rand(1:length(q))  # random exploration
    else
        rand(findall(==(maximum(q)), q)) # greedy exploitation with ties broken
    end
end

# ╔═╡ 95ee4cf6-afd8-4979-b907-10d13aa3b079
md"# Markov Decision Processes

## A Side-Remark about Fixed-Point Iteration

Some equations can be nicely solved by a fixed point iteration.

A simple example is the Babylonian method for computing the square root.
If we transform the equation ``x^2 = a`` to ``2x^2 = a + x`` and finally to ``x = \frac12\left(\frac{a}x + x\right) = f_a(x)`` we get an equation ``x = f_a(x)`` that can be solved by fixed point iteration, i.e. we start with an arbitrary ``x^{(0)} > 0`` and compute iteratively ``x^{(k)} = f_a(x^{(k-1)})`` until ``x^{(k)} \approx x^{(k-1)}``.
"

# ╔═╡ a09a0468-645d-4d24-94ef-feb4822cf2b2
function fixed_point_iteration(f, x0)
    old_x = copy(x0)
    while true
        new_x = f(old_x)
        new_x ≈ old_x && break # break, if the values does not change much anymore
		old_x = copy(new_x)
    end
    old_x
end

# ╔═╡ 59407024-c9ff-4b0d-b4fb-aa295654b5b0
fixed_point_iteration(x -> 1/2*(2/x + x), 10.)

# ╔═╡ 0567c424-e98e-46fc-8508-f53df44d5fc7
md"We see that our fixed point iteration gives the same result as ``\sqrt2 = ``$(sqrt(2)) (up to the precision set by ≈).

Not every equation has this property that it can be used for a fixed point iteration (e.g. ``x = \frac{a}x`` would not work for computing the square root), but the Bellman equation *is* of this type.

## Iterative Policy Evalution
We will now use an iterative procedure to compute the Q-values of an arbitrary policy."

# ╔═╡ aca82f30-ac46-4b41-bf01-c824859567bf
function evaluate_policy(policy, T, R)
    nₐ, nₛ = size(T)
    r = mean.(R)' # this is an array of (nₐ, nₛ) average reward values
    fixed_point_iteration(Q -> r .+ [sum(Q[policy(s′), s′]*T[a, s][s′] for s′ in 1:nₛ)
                                     for a in 1:nₐ, s in 1:nₛ],
                          zeros(nₐ, nₛ))
end;

# ╔═╡ 9e02b30a-ef38-4495-bfcb-6bb2ab838230
begin
    struct DeterministicPolicy
        # the i'th entry of this vector is the action for state i.
        actions::Vector{Int}
    end
    (policy::DeterministicPolicy)(s) = policy.actions[s]
end

# ╔═╡ 269c929e-fea1-4bf9-bd68-bb52b9c965df
md"In the last line above we made objects of type `DeterministicPolicy` callable (see e.g. [here](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects))."

# ╔═╡ 4b1055ce-5ae8-4ee2-ab9b-ac68630d1deb
policy1 = DeterministicPolicy([1, 2, 1, 2, 1, 2, 1]);

# ╔═╡ 8486e9c5-0db8-4868-9898-cfb752d4b8f8
policy1(3)

# ╔═╡ eeff23de-a926-4e66-abcb-370fdd577c3c
md"This allows us now to evaluate `policy1` on the transition dynamics T and reward probabilites R of our chase au trésor."

# ╔═╡ 3ab57bc9-907a-4b16-ae20-1e1cf2536e38
showQ(evaluate_policy(policy1, T, R))

# ╔═╡ 11141e67-421a-4e02-a638-f03b47ffb53c
md"## Policy Iteration

The idea of policy iteration is to start with a random deterministic policy ``\pi^{(0)}(s)``, compute ``Q_{\pi^{(0)}}(s, a)`` with iterative policy evaluation and then update the policy such that ``\pi^{(1)}(s) = \arg\max_aQ_{\pi^{(0)}}(s, a)``. Policy evaluation and policy update are iterated until the policy does not change anymore in successive iterations, i.e. ``\pi^{(k)}(s) = \pi^{(k-1)}(s)``, and we have found the optimal policy.
"

# ╔═╡ 5be36d3c-db37-4566-a469-d1a793d26a87
function policy_iteration(T, R)
    nₐ, nₛ = size(T)
    policy = DeterministicPolicy(rand(1:nₐ, nₛ))
    while true
        Q = evaluate_policy(policy, T, R)
        policy_new = DeterministicPolicy([argmax(Q[:, s]) for s in 1:nₛ])
        policy_new.actions == policy.actions && break
        policy = policy_new
    end
    policy
end


# ╔═╡ 5c294e67-3590-41e1-bf40-b1bcc922f57a
optimal_policy = policy_iteration(T, R)

# ╔═╡ e74a4a44-ebe5-4596-b08e-d3caeb426f1c
showQ(evaluate_policy(optimal_policy, T, R))

# ╔═╡ f9fa7e6a-f1b9-428e-a3b2-044b26b96965
md"We see that the optimal policy is to open the right door in the first (red) room
and then open the right door if there is no guard and the left door otherwise.
Although there is less often a guard in the blue room, there is reward to get there.
"

# ╔═╡ dc7c5cb0-30b2-4427-8aeb-f312a88effd1
md"# Q-Learning

Q-learning can be seen as a generalized policy iteration method where the transition probabilities and the reward probabilities are unknown. Instead of running full iterative policy evaluation between each step of policy update, Q-learning updates only the Q-value of the state-action pair experienced in the last time step.
"

# ╔═╡ e39227c8-8148-43cb-9351-774682b65646
begin
    Base.@kwdef struct QLearner # this defines a type QLearner
        Q = zeros(size(T)...)      # default initial Q-values
        λ = 0.01
    end
	"""
	    update!(l::QLearner, a, s, r)

	Updates an `MLClearner` object `l` for a given
	state `s`, action `a`, reward `r` and next state `s′`.
	"""
    function update!(l::QLearner, s, a, r, s′)
        Q, λ = l.Q, l.λ
        δ = (r + maximum(Q[:, s′]) - Q[a, s])
        Q[a, s] = Q[a, s] + λ * δ
        l
    end
	"""
	    update!(l::QLearner, episode)

	Updates an `QLearner` oject `l` for a full episode.
	"""
    function update!(l::QLearner, episode)
		length(episode) > 0 || return l
        s, a, r, s′ = last(episode)
        update!(l, s, a, r, s′)
        l
    end
    reset!(l::QLearner) = l.Q .= 0
    qlearner = QLearner()
end;

# ╔═╡ bf600a9b-484e-4a7b-bf3b-409ebad51cd0
md"""
If you want to see Q-learning in action, choose "qlearner" below and play our
chase au trésor. The displayed Q-values will then be for the `qlearner`.

Learner $(@bind learner Select(["mclearner", "qlearner"]))

Below you see the Q-values of an epsilon-greedy policy played for ``10^6`` steps.
Note that is comes close to the solution found with policy iteration. This is different than what we found with the Monte Carlo Learner. The reason is that the Monte Carlo Learner estimates the Q-values for the epsilon-greedy policy, whereas the Q-Learner estimates the Q-values for the greedy policy, even though the actions are choosen according to the epsilon-greedy policy. The Monte Carlo Learner is called *on-policy* because it evaluates the policy used for action selection. The Q-Learner is called *off-policy*, because it evaluates the greedy policy, while selecting actions with the epsilon-greedy policy.
"""

# ╔═╡ 5a705ba7-dd9c-411c-a910-659fb1ec9f82
md"Below you see the Q-values of the \`$learner\`."

# ╔═╡ e876c526-30f9-458d-abf5-e20e6aa0268e
let
    do_it
    if learner == "mclearner"
        showQ(mclearner.Q)
    else
        showQ(qlearner.Q)
    end
end

# ╔═╡ 5dd76aaa-4aca-4fa3-b85c-0578cb178560
md"""# Q-Learning with Function Approximation

Discrete state representations, where each state has its own state number (e.g. "blue room with guard" = state 3), are fine for small problems, but in many situations the number of different states is too large to be enumerated with limited memory. Furthermore, with discrete state representations it is impossible to generalize between states. E.g. we may quickly learn that it is not a good idea to open a door with a guard in front of it, independently of the room color and the position of the door. With a discrete representation it is impossible to learn such generalizations, because state 3 is as different from state 5 as it is from state 4 (without additional information). Therefore it is often desirable to use a distributed representation of the states, where the different elements of the input vector indicate the presence of absence of certain features.
"""

# ╔═╡ 52001dae-6e94-4d80-89b0-8fda1a37a0a6
function distributed_representation(s)
    [s == 1; # 1 if in room 1
     s ∈ (2, 3); # 1 if in room 2
     s ∈ (4, 5); # 1 if in room 3
     s == 6; # 1 if in treasure room
     s == 7; # 1 if KO
     s ∈ (3, 5)] # 1 if guard present
end

# ╔═╡ 31da392f-0283-4a50-adfa-ac1c14ad2ac3
distributed_representation(5)

# ╔═╡ b06097e9-ef1e-4a18-839e-9e3758e5201b
begin
    Base.@kwdef mutable struct DeepQLearner # this defines a type DeepQLearner
        Qnetwork
        optimizer = ADAM()
    end
	"""
	    update!(l::DeepQLearner, a, s, r)

	Updates an `MLClearner` object `l` for a given
	state `s`, action `a`, reward `r` and next state `s′`.
	"""
    function update!(l::DeepQLearner, s, a, r, s′)
        Qnetwork = l.Qnetwork
        x = distributed_representation(s)
        x′ = distributed_representation(s′)
        Q′ = maximum(Qnetwork(x′))
        θ = params(Qnetwork)
        gs = gradient(θ) do
            (r + Q′ - Qnetwork(x)[a])^2
        end
        Flux.update!(l.optimizer, θ, gs)
        l
    end
	"""
	    update!(l::DeepQLearner, episode)

	Updates an `DeepQLearner` oject `l` for a full episode.
	"""
    function update!(l::DeepQLearner, episode)
		length(episode) > 0 || return l
        s, a, r, s′ = last(episode)
        update!(l, s, a, r, s′)
        l
    end
    reset!(l::DeepQLearner) = nothing # not implemented
end;

# ╔═╡ e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
let
    do_it
    _learner = if learner == "mclearner"
        mclearner
    else
        qlearner
    end
    update!(_learner, game_state.episode_recorder)
    if game_state.action == "reset episode"
        reset!(game_state)
    elseif game_state.action == "open left door"
        act!(game_state, 1)
    elseif game_state.action == "open right door"
        act!(game_state, 2)
    end
    if game_state.action == "reset learner"
        reset!(_learner)
    end
    d = distributed_representation(game_state.state)
    room = findfirst(d)
    guard = d[6]
    guard_door = room > 2
    gold = d[4]
    room_colors = [:red, :blue, :green, :orange, :black]
    plot(xlim = (0, 1), ylim = (0, 1),
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
        r = floor(Int, game_state.reward)
        x = x[1:r]
        y = y[1:r]
        scatter!(x, y, markerstrokewidth = 3, c = :yellow, markersize = 28)
    end
    if room == 5
        annotate!([(.5, .5, "K.O.", :red)])
    end
    rs = length(game_state.episode_recorder) == 0 ? [0] : getindex.(game_state.episode_recorder, 3)
    annotate!([(.5, .9, "reward = $(game_state.reward)", :white),
               (.5, .8, "cumulative reward = $(join(rs, " + ")) = $(sum(rs))", :white)
              ])
end

# ╔═╡ 712c2a9e-4413-4d7a-b729-cfb219723256
let mclearner = MCLearner(),
    game_state = GameState()
    for episode in 1:10^5
        for steps in 1:2
            s = game_state.state
            a = epsilon_greedy_policy(mclearner.Q[:, s])
            act!(game_state, a)
        end
        update!(mclearner, game_state.episode_recorder)
        reset!(game_state)
    end
    showQ(mclearner.Q)
end

# ╔═╡ bd8557bc-86f2-4ccc-93e9-a6bd843e80be
let qlearner = QLearner(),
    game_state = GameState()
    for episode in 1:10^6
        for steps in 1:2
            s = game_state.state
            a = epsilon_greedy_policy(qlearner.Q[:, 2])
            act!(game_state, a)
            update!(qlearner, game_state.episode_recorder)
        end
        reset!(game_state)
    end
    showQ(qlearner.Q)
end

# ╔═╡ af7015c4-e7ab-4e18-bd37-ccffe4ec2928
dql = let deepqlearner = DeepQLearner(Qnetwork = Chain(Dense(6, 10, relu),
                                                       Dense(10, 2))),
    game_state = GameState()
    for episode in 1:10^5
        for steps in 1:2
            x = distributed_representation(game_state.state)
            a = epsilon_greedy_policy(deepqlearner.Qnetwork(x))
            act!(game_state, a)
            update!(deepqlearner, game_state.episode_recorder)
        end
        reset!(game_state)
    end
    deepqlearner
end;

# ╔═╡ ce334e27-9b66-4692-becd-cfc24ff58cb1
showQ(hcat([dql.Qnetwork(distributed_representation(s)) for s in 1:7]...))

# ╔═╡ 6464f6d3-2fcf-493f-ad21-206d9c273a16
md" The Q-values of our DeepQLearner are not the same as in the discrete representation.
For example, all those elements that should be zero are clearly non-zero. However, for decision making all the matters are the differences between the Q-values of action left and right in each state, and these differences match rather well (except for the terminal states where no decisions are taken anyway)."

# ╔═╡ 3a643502-7d78-4d0c-a53f-913f35306258
[argmax(dql.Qnetwork(distributed_representation(s))) for s in 1:7]

# ╔═╡ aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
md"# Learning to Play Tic-Tac-Toe

In the following example we will learn to play Tic-Tac-Toe with a Monte Carlo Learner.
The Tic-Tac-Toe environment is loaded from the [`ReinforcementLearning` package](https://juliareinforcementlearning.org/). This package also provides multiple helper functions to interact with this environment, like `state`, `is_terminated`, `legal_action_space`, `ReinforcementLearning.reward`, `ReinforcementLearning.reset!` etc.
"

# ╔═╡ ae72b2c3-fe7b-4ffb-b45c-e589529209c7
tictactoe = TicTacToeEnv();

# ╔═╡ 159d071e-3cd5-4007-9d78-db340846f97f
state(tictactoe) # loads the state as a text representation

# ╔═╡ ddd02bd9-9577-44a8-8bb6-2b1d11938121
state(tictactoe, Observation{Int}()) # loads the state as an integer

# ╔═╡ d094b8d7-5cc7-4bc0-a406-2e3c85f9fb60
md"To perform an action we can simply call the environment with an integer from 1 to 9.
1 means placing a cross `x` or nought `o` (depending on whose turn it is) at the top left corner, 3 is at the bottom left corner, 7 top right and 9 bottom right."

# ╔═╡ aca72848-c488-46d1-a3df-0cf6fb04f4a3
let
	tictactoe(5) # place a cross in the center
	Text(state(tictactoe)) # diplay the state nicely
end

# ╔═╡ 4b90f08e-0183-4f8b-a6cd-e66b6938f4c8
legal_action_space(tictactoe) # now action 5 is no longer available

# ╔═╡ 86a78734-2b31-4ee5-8560-8a9388672b45
ReinforcementLearning.reset!(tictactoe);

# ╔═╡ 59029032-1c91-4da1-a61b-6a56449dcd2c
md"Let us now actually play the game. You can choose different modes below to play against yourself (or another human player), against the computer (trained in self-play with our `MCLearner`; see below) or computer against computer. To advance the game when the computer plays against itself you have to use the `step` button."

# ╔═╡ b257795e-d0d6-495a-ae11-dca090ff786a
md"""

action $(@bind tact Select(string.(1:9))) $(@bind tres Button("reset game")) $(@bind tstep Button("step"))"""

# ╔═╡ 4a9fb8a0-81fb-4e59-8208-61df1dbd8255
md"""Player Cross: $(@bind player1 Select(["human", "machine"])) Player Nought: $(@bind player2 Select(["human", "machine"]))"""

# ╔═╡ d99a218a-56e8-4081-bd1a-ba7729f529cf
md"The computer player is using a greedy policy with Q-values learned in self-play with the code below."

# ╔═╡ 6a96c33a-b6b3-4a0a-83c8-a0df113887d0
mcl = let
    nₛ = length(state_space(tictactoe, Observation{Int}()))
    mcl = MCLearner(N = zeros(Int, 9, nₛ),
                    Q = zeros(9, nₛ))
    episode_cross = Tuple{Int, Int, Float64, Nothing}[]
    episode_nought = Tuple{Int, Int, Float64, Nothing}[]
    for game in 1:10^5
        ReinforcementLearning.reset!(tictactoe)
        move = 0
        while true
            move += 1
            legal_a = legal_action_space(tictactoe)
            s = state(tictactoe, Observation{Int}())
            i = epsilon_greedy_policy(mcl.Q[legal_a, s])
            a = legal_a[i]
            tictactoe(a)
            if is_terminated(tictactoe)
                r = ReinforcementLearning.reward(tictactoe) # reward of next player
                if isodd(move)
                    r *= -1 # cross finished
                end
                push!(episode_cross, (s, a, r, nothing))
                push!(episode_nought, (s, a, -r, nothing))
                break
            else
                push!(isodd(move) ? episode_cross : episode_nought,
                      (s, a, 0., nothing))
            end
        end
        update!(mcl, episode_cross)
        update!(mcl, episode_nought)
        empty!(episode_cross)
        empty!(episode_nought)
    end
    mcl
end;

# ╔═╡ 0b145605-f75f-47ee-ab64-ab77eb812b67
begin tres
    ReinforcementLearning.reset!(tictactoe)
    _tres = rand()
end;

# ╔═╡ 7f3e31e5-5f5e-4188-aeb4-01d30a6dc26f
begin
    a = parse(Int, tact)
    if a in legal_action_space(tictactoe)
        tictactoe(a)
    end
end;

# ╔═╡ c692cc6e-dbb5-40e9-aeaa-486b098c3af1
begin
    import ReinforcementLearning.ReinforcementLearningEnvironments: Cross, Nought
    tres, a, tstep
    a2 = nothing
    function autoplaying(player1, player2, tictactoe)
        is_terminated(tictactoe) && return false
        player1 == "machine" && player2 == "machine" && return true
        player1 == "machine" && current_player(tictactoe) == Cross() && return true
        player2 == "machine" && current_player(tictactoe) == Nought() && return true
        false
    end
    if autoplaying(player1, player2, tictactoe)
        legal_a = legal_action_space(tictactoe)
        a2 = legal_a[argmax(mcl.Q[legal_a, state(tictactoe, Observation{Int}())])]
        tictactoe(a2)
    end
end;


# ╔═╡ ebebd97a-9dc2-4b39-a998-9279d52c57e5
let _tres, a, a2
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
    s *= "\n\n" * state(tictactoe)
    s *= "\n\n Legal actions:\n\n$(join([join([k in legal_a ? string(k) : " " for k in 3 .* (0:2) .+ j], " ") for j in 1:3], "\n"))"
    s *= "\n\n Game state: $(state(tictactoe, Observation{Int}()))"
    Text(s)
end

# ╔═╡ b6b835d1-8f84-4148-8d5b-c7aea6b0c312
Markdown.parse("""# Exercises
## Conceptual
1. Consider an agent that experiences episode 1: ``((S_1 = s_1, A_1 = a_1, R_1 = 2), (S_2 = s_7, A_2 = a_2, R_2 = 1), (S_3 = s_2, A_3 = a_1, R_3 = -4))`` and episode 2: ``((S_1 = s_1, A_1 = a_1, R_1 = 1)), (S_2 = s_6, A_2 = a_1, R_2 = -1), (S_3 = s_2, A_3 = a_2, R_3 = 1))``.
    1. Compute ``Q(s_1, a_1)`` with Monte Carlo Estimation.
    2. Compute ``Q(s_1, a_1)`` with Q-Learning, learning rate ``\\lambda = 0.5`` and initial ``Q(s_1, a_1) = 0``.
2. Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as different from each other as possible. The framework is abstract and flexible and can be applied in many different ways. Stretch its limits in some way in at least one of your examples.
3. Show for infinite-horizon problems with discount factor ``0 < \\gamma < 1`` the Bellman equations for a determinstic policy ``\\pi`` are given by ``Q_\\pi(s, a) = \\bar r(s, a) + \\gamma \\sum_{s'}P(s'|s, a)Q_\\pi(s', \\pi(s'))``. Start with the definition ``Q_\\pi(S_t, A_t) = \\mathrm{E}\\left[R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\gamma^3R_{t+4} + \\cdots \\right]`` and show all the intermediate steps that lead to the result.
## Applied
1. Consider driving a race car around a turn like those shown in the figure below. You want to go as fast as possible, but not so fast as to run off the track. In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step. The actions are increments to the velocity components. Each may be changed by +1, 1, or 0 in each step, for a total of nine (3 x 3) actions. Both velocity components are restricted to be nonnegative and less than 5, and they cannot both be zero except at the starting line. Each episode begins in one of the randomly selected start states with both velocity components zero and ends when the car crosses the finish line. The rewards are 1 for each step until the car crosses the finish line. If the car hits the track boundary, it is moved back to a random position on the starting line, both velocity components are reduced to zero, and the episode continues. Before updating the car’s location at each time step, check to see if the projected path of the car intersects the track boundary. If it intersects the finish line, the episode ends; if it intersects anywhere else, the car is considered to have hit the track boundary and is sent back to the starting line. To make the task more challenging, with probability 0.1 at each time step the velocity increments are both zero, independently of the intended increments. Use an epsilon-greedy policy with decreasing epsilon and Monte Carlo Estimation of the Q-values to find the optimal policy from each starting state. Exhibit several trajectories following the optimal policy (but turn the noise off for these trajectories).

$(MLCourse.embed_figure("carrace.png"))

2. Consider the task of driving an underpowered car up a steep mountain road, as suggested in the figure below.  The difficulty is that gravity is stronger than the car’s engine, and even at full throttle the car cannot accelerate up the steep slope. The only solution is to first move away from the goal and up the opposite slope on the left. Then, by applying full throttle the car can build up enough inertia to carry it up the steep slope even though it is slowing down the whole way. This is a simple example of a continuous control task where things have to get worse in a sense (farther from the goal) before they can get better.  Many control methodologies have great difficulties with tasks of this kind unless explicitly aided by a human designer.  The reward in this problem is 1 on all time steps until the car moves past its goal position at the top of the mountain, which ends the episode. There are three possible actions: full throttle forward (+1), full throttle reverse (-1), and zero throttle (0). The car moves according to a simplified physics. You can load this environment as `mountain_car_env = MountainCarEnv()`. Use Q-Learning with Function Approximation to find a good policy for this problem.

$(MLCourse.embed_figure("mountaincar.png"))
""")

# ╔═╡ Cell order:
# ╟─ce405f97-6d60-4ae4-b183-79e6c88d9811
# ╟─e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
# ╟─12697628-d520-4c3a-9e33-f7c8e17f0dea
# ╟─86a4f1ef-c81c-4888-95ed-15269d38fbea
# ╟─5a705ba7-dd9c-411c-a910-659fb1ec9f82
# ╟─e876c526-30f9-458d-abf5-e20e6aa0268e
# ╟─761d690d-5c73-40dd-b38c-5af67ee837c0
# ╟─2b7329a7-177a-468d-b9b9-1cba648ba570
# ╟─c50abfb9-6f0a-4a80-a54d-cd04e7f99aa3
# ╟─b0b89b30-38f1-45c1-8c7a-796ea2f41e8d
# ╠═7c756af7-08a0-4878-8f53-9eaa79c16c1a
# ╟─fe5ae6b8-bc5e-453a-96fd-f3e7dab135e5
# ╟─6dced47e-cc0f-4ae5-bdd0-a551c6d5a5b3
# ╠═64863a37-1c25-4d1c-9f2d-33d87b4039a3
# ╠═712c2a9e-4413-4d7a-b729-cfb219723256
# ╟─95ee4cf6-afd8-4979-b907-10d13aa3b079
# ╠═a09a0468-645d-4d24-94ef-feb4822cf2b2
# ╠═59407024-c9ff-4b0d-b4fb-aa295654b5b0
# ╟─0567c424-e98e-46fc-8508-f53df44d5fc7
# ╠═aca82f30-ac46-4b41-bf01-c824859567bf
# ╠═9e02b30a-ef38-4495-bfcb-6bb2ab838230
# ╟─269c929e-fea1-4bf9-bd68-bb52b9c965df
# ╠═4b1055ce-5ae8-4ee2-ab9b-ac68630d1deb
# ╠═8486e9c5-0db8-4868-9898-cfb752d4b8f8
# ╟─eeff23de-a926-4e66-abcb-370fdd577c3c
# ╠═3ab57bc9-907a-4b16-ae20-1e1cf2536e38
# ╟─11141e67-421a-4e02-a638-f03b47ffb53c
# ╠═5be36d3c-db37-4566-a469-d1a793d26a87
# ╠═5c294e67-3590-41e1-bf40-b1bcc922f57a
# ╠═e74a4a44-ebe5-4596-b08e-d3caeb426f1c
# ╟─f9fa7e6a-f1b9-428e-a3b2-044b26b96965
# ╟─dc7c5cb0-30b2-4427-8aeb-f312a88effd1
# ╠═e39227c8-8148-43cb-9351-774682b65646
# ╟─bf600a9b-484e-4a7b-bf3b-409ebad51cd0
# ╠═bd8557bc-86f2-4ccc-93e9-a6bd843e80be
# ╟─5dd76aaa-4aca-4fa3-b85c-0578cb178560
# ╠═52001dae-6e94-4d80-89b0-8fda1a37a0a6
# ╠═31da392f-0283-4a50-adfa-ac1c14ad2ac3
# ╠═26918860-b3c3-454a-8767-2116b2ece629
# ╠═b06097e9-ef1e-4a18-839e-9e3758e5201b
# ╠═af7015c4-e7ab-4e18-bd37-ccffe4ec2928
# ╠═ce334e27-9b66-4692-becd-cfc24ff58cb1
# ╟─6464f6d3-2fcf-493f-ad21-206d9c273a16
# ╠═3a643502-7d78-4d0c-a53f-913f35306258
# ╟─aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
# ╠═1a17e9b2-a439-4521-842c-96ebe0378919
# ╠═ae72b2c3-fe7b-4ffb-b45c-e589529209c7
# ╠═159d071e-3cd5-4007-9d78-db340846f97f
# ╠═ddd02bd9-9577-44a8-8bb6-2b1d11938121
# ╟─d094b8d7-5cc7-4bc0-a406-2e3c85f9fb60
# ╠═aca72848-c488-46d1-a3df-0cf6fb04f4a3
# ╠═4b90f08e-0183-4f8b-a6cd-e66b6938f4c8
# ╠═86a78734-2b31-4ee5-8560-8a9388672b45
# ╟─59029032-1c91-4da1-a61b-6a56449dcd2c
# ╟─b257795e-d0d6-495a-ae11-dca090ff786a
# ╟─ebebd97a-9dc2-4b39-a998-9279d52c57e5
# ╟─4a9fb8a0-81fb-4e59-8208-61df1dbd8255
# ╟─d99a218a-56e8-4081-bd1a-ba7729f529cf
# ╠═6a96c33a-b6b3-4a0a-83c8-a0df113887d0
# ╟─0b145605-f75f-47ee-ab64-ab77eb812b67
# ╟─7f3e31e5-5f5e-4188-aeb4-01d30a6dc26f
# ╟─c692cc6e-dbb5-40e9-aeaa-486b098c3af1
# ╟─b6b835d1-8f84-4148-8d5b-c7aea6b0c312
# ╟─8bd459cb-20bb-483e-a849-e18caae3beef
# ╟─b97724e4-d7b0-4085-b88e-eb3c5bcbe441
