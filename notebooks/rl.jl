### A Pluto.jl notebook ###
# v0.19.27

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
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse", "RLEnv"))
end
using StatsBase, DataFrames, Plots, ReinforcementLearning, MLCourse
end

# ╔═╡ b3793299-c916-40d3-bd87-31153fc3781a
using PlutoUI, PlutoUI.BuiltinsNotebook.HypertextLiteral; PlutoUI.TableOfContents()

# ╔═╡ bb299fa3-925f-4f12-89a0-aa890bf56c25
using Flux # used to define the Qnetwork as a feedforward neural network

# ╔═╡ ce405f97-6d60-4ae4-b183-79e6c88d9811
md"# 1. Chasse au trésor"

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

	Updates an `MCLearner` object `l` for a given
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

	Updates an `MCLearner` object `l` for a full episode.
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
md"Now we let this agent play the game for 10^6 episodes and learn the Q-values with Monte Carlo estimation."


# ╔═╡ 95ee4cf6-afd8-4979-b907-10d13aa3b079
md"# Markov Decision Processes

In this section we assume that all the transition probabilities and all rewards are known. Then the only problem to solve is to find in each state the optimal policy. This optimal policy can be found with a fixed-point iteration called \"policy iteration\", where one starts with an arbitrary policy and iterates a policy improvement step until the optimal policy is found.

## A Side-Remark about Fixed-Point Iteration

Some equations can be nicely solved by a fixed point iteration.

A simple example is the Babylonian method for computing the square root.
If we transform the equation ``x^2 = a`` to ``2x^2 = a + x^2`` and finally to ``x = \frac12\left(\frac{a}x + x\right) = f_a(x)`` we get an equation ``x = f_a(x)`` that can be solved by fixed point iteration, i.e. we start with an arbitrary ``x^{(0)} > 0`` and compute iteratively ``x^{(k)} = f_a(x^{(k-1)})`` until ``x^{(k)} \approx x^{(k-1)}``.
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


# ╔═╡ f9fa7e6a-f1b9-428e-a3b2-044b26b96965
md"We see that the optimal policy is to open the right door in the first (red) room
and then open the right door if there is no guard and the left door otherwise.
Although there is less often a guard in the blue room, there is less reward to get there.
"

# ╔═╡ dc7c5cb0-30b2-4427-8aeb-f312a88effd1
md"# Q-Learning

Q-learning is an alternative method for updating Q-values that relies on the idea that ``Q(s_t, a_t)`` for the optimal policy can be estimated by knowing ``r_t`` and ``\max_a Q(s_{t+1}, a)``. In Q-learning the transition probabilities and rewards are unknown and the optimal policy has to be found while exploring the environment.
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

	Updates a `QLearner` object `l` for a given
	state `s`, action `a`, reward `r` and next state `s′`.
	"""
    function update!(learner::QLearner, s, a, r, s′)
        Q, λ = learner.Q, learner.λ
        δ = (r + maximum(Q[:, s′]) - Q[a, s])
        Q[a, s] = Q[a, s] + λ * δ
        learner
    end
    reset!(learner::QLearner) = learner.Q .= 0
end;

# ╔═╡ 5dd76aaa-4aca-4fa3-b85c-0578cb178560
md"""# Deep Q-Learning

Discrete state representations, where each state has its own state number (e.g. "blue room with guard" = state 3), are fine for small problems, but in many situations the number of different states is too large to be enumerated with limited memory. Furthermore, with discrete state representations it is impossible to generalize between states. E.g. we may quickly learn that it is not a good idea to open a door with a guard in front of it, independently of the room color and the position of the door. With a discrete representation it is impossible to learn such generalizations, because state 3 is as different from state 5 as it is from state 4 (without additional information). Therefore it is often desirable to use a distributed representation of the states, where the different elements of the input vector indicate the presence of absence of certain features.
"""

# ╔═╡ c16fbc62-3dcc-4097-bc82-ec26d89794ec
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
        θ = Flux.params(Qnetwork)
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
end;

# ╔═╡ 6464f6d3-2fcf-493f-ad21-206d9c273a16
md" The Q-values of our DeepQLearner are not the same as in the discrete representation.
For example, all those elements that should be zero are clearly non-zero. However, for decision making all that matters are the differences between the Q-values of action left and right in each state, and these differences match rather well (except for the terminal states where no decisions are taken anyway)."

# ╔═╡ aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
md"# Learning to Play Tic-Tac-Toe

In the following example we will learn to play Tic-Tac-Toe with a Monte Carlo Learner.
The Tic-Tac-Toe environment is loaded from the [`ReinforcementLearning` package](https://juliareinforcementlearning.org/). This package also provides multiple helper functions to interact with this environment, like `state`, `is_terminated`, `legal_action_space`, etc.
"

# ╔═╡ 1a17e9b2-a439-4521-842c-96ebe0378919
import ReinforcementLearning: TicTacToeEnv, legal_action_space, current_player, RLBase

# ╔═╡ ae72b2c3-fe7b-4ffb-b45c-e589529209c7
tictactoe = TicTacToeEnv();

# ╔═╡ d094b8d7-5cc7-4bc0-a406-2e3c85f9fb60
md"To perform an action we can simply call the environment with an integer from 1 to 9.
1 means placing a cross `x` or nought `o` (depending on whose turn it is) at the top left corner, 3 is at the bottom left corner, 7 top right and 9 bottom right."

# ╔═╡ 4b90f08e-0183-4f8b-a6cd-e66b6938f4c8
legal_action_space(tictactoe) # now action 5 is no longer available

# ╔═╡ 59029032-1c91-4da1-a61b-6a56449dcd2c
md"Let us now actually play the game. You can choose different modes below to play against yourself (or another human player), against the computer (trained in self-play with our `MCLearner`; see below) or computer against computer. To advance the game when the computer plays against itself you have to use the `step` button."

# ╔═╡ d99a218a-56e8-4081-bd1a-ba7729f529cf
md"The computer player is using a greedy policy with Q-values learned in self-play with the code below."

# ╔═╡ e61c6d43-a097-4f44-a343-05371b4932ef
begin
    is_terminated(env::TicTacToeEnv) = RLBase.is_terminated(env)
    act!(env::TicTacToeEnv, a) = env(a)
    reset!(env::TicTacToeEnv) = RLBase.reset!(env)
    reward(env::TicTacToeEnv) = RLBase.reward(env)
    state(env::TicTacToeEnv) = RLBase.state(env, RLBase.Observation{Int}())
end;

# ╔═╡ b6b835d1-8f84-4148-8d5b-c7aea6b0c312
md"""# Exercises
## Conceptual

Consider an agent that experiences episode 1:

```math
\begin{align*}
S_1 &= s_1, A_1 = a_1, R_2 = 2,\\
S_2 &= s_7, A_2 = a_2, R_3 = 1,\\
S_3 &= s_6, A_3 = a_2, R_4 = 4,\\
S_4 &= s_3, A_4 = a_1, R_5 = 0
\end{align*}
```

and episode 2:

```math
\begin{align*}
S_1 &= s_1, A_1 = a_1, R_2 = 1,\\
S_2 &= s_6, A_2 = a_1, R_3 = -1,\\
S_3 &= s_2, A_3 = a_2, R_4 = 1
\end{align*}
```.

(a) Compute ``Q(s_1, a_1)`` with Monte Carlo Estimation.

(b) Compute ``Q(s_1, a_1)`` with Q-Learning, learning rate ``\lambda = 0.5`` and initial ``Q(s, a) = 0`` for all ``s`` and ``a``.

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
    Base.@kwdef mutable struct CliffWalkingEnv <: RLBase.AbstractEnv
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


# ╔═╡ 8bd459cb-20bb-483e-a849-e18caae3beef
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 412d8fcb-8f98-43b6-9235-a4c228317427
MLCourse.footer()

# ╔═╡ d88c1d9b-3396-42a2-8ebd-81851f778602
begin
    struct CounterButtons
        labels
    end

	function Base.show(io::IO, m::MIME"text/html", button::CounterButtons)
		buttons = [@htl("""<input type="button" value="$label"> """)
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

    Base.get(::CounterButtons) = [0, 0]
	PlutoUI.BuiltinsNotebook.Bonds.initial_value(::CounterButtons) = [0, 0]
	PlutoUI.BuiltinsNotebook.Bonds.possible_values(::CounterButtons) = PlutoUI.BuiltinsNotebook.Bonds.InfinitePossibilities()
	function PlutoUI.BuiltinsNotebook.Bonds.validate_value(::CounterButtons, val)
		val isa Vector{Int} && val[1] >= 0 && val[2] >= 0
	end
end

# ╔═╡ 7cf26d6c-5a67-4bd6-8ef2-34559b53685b
@bind chasse_actions CounterButtons(["open left door", "open right door",
                                     "reset episode", "reset learner"])

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

# ╔═╡ 94c0859f-f2d4-4b00-9793-14c823bbc705
@bind tictactoe_action CounterButtons([string.(1:9); "reset game"; "step"])

# ╔═╡ 4a9fb8a0-81fb-4e59-8208-61df1dbd8255
md"""Player Cross: $(@bind player1 Select(["human", "machine"])) Player Nought: $(@bind player2 Select(["human", "machine"]))"""

# ╔═╡ 8c0a5e46-c790-4cec-ac57-1b2813b81358
@bind cw_action CounterButtons(["left", "right", "up", "down", "reset"])

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
#     chasse = ChasseAuTresorEnv()
end;

# ╔═╡ 712c2a9e-4413-4d7a-b729-cfb219723256
let mclearner = MCLearner(na = 2, ns = 7),
    chasse = ChasseAuTresorEnv()
    for _ in 1:10^6
		reset!(chasse)
        episode = []
        for steps in 1:2
            s = state(chasse) # observe the environment
            a = epsilon_greedy_policy(mclearner.Q[:, s]) # use policy to find action
            act!(chasse, a) # act in the environment
            r = reward(chasse) # observe the reward
            push!(episode, (s, a, r)) # store in the episode buffer
        end
        update!(mclearner, episode) # update learner after the end of the episode
    end
    showQ(mclearner.Q)
end

# ╔═╡ 3ab57bc9-907a-4b16-ae20-1e1cf2536e38
showQ(evaluate_policy(policy1, T, R))

# ╔═╡ 5c294e67-3590-41e1-bf40-b1bcc922f57a
optimal_policy = policy_iteration(T, R)

# ╔═╡ e74a4a44-ebe5-4596-b08e-d3caeb426f1c
showQ(evaluate_policy(optimal_policy, T, R))

# ╔═╡ bd8557bc-86f2-4ccc-93e9-a6bd843e80be
let qlearner = QLearner(na = 2, ns = 7),
    chasse = ChasseAuTresorEnv()
    for _ in 1:10^6
        reset!(chasse)
        for steps in 1:2
            s = state(chasse) # observe the environment
            a = epsilon_greedy_policy(qlearner.Q[:, s]) # use policy to find action
            act!(chasse, a) # act in the environment
            r = reward(chasse) # observe the reward
            s′ = state(chasse) # observe the next state
            update!(qlearner, s, a, r, s′) # update the learner
        end
    end
    showQ(qlearner.Q)
end

# ╔═╡ af7015c4-e7ab-4e18-bd37-ccffe4ec2928
dql = let deepqlearner = DeepQLearner(Qnetwork = Chain(Dense(6, 10, relu),
                                                       Dense(10, 2))),
    chasse = ChasseAuTresorEnv()
    for episode in 1:10^5
        for steps in 1:2
            x = distributed_representation(chasse.state)
            a = epsilon_greedy_policy(deepqlearner.Qnetwork(x))
            act!(chasse, a)
            update!(deepqlearner, chasse.episode_recorder)
        end
        reset!(chasse)
    end
    deepqlearner
end;

# ╔═╡ 3a643502-7d78-4d0c-a53f-913f35306258
[argmax(dql.Qnetwork(distributed_representation(s))) for s in 1:7]

# ╔═╡ ce334e27-9b66-4692-becd-cfc24ff58cb1
showQ(hcat([dql.Qnetwork(distributed_representation(s)) for s in 1:7]...))

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
mcl = let tictactoe = TicTacToeEnv()
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

# ╔═╡ 37d0f305-3aea-4e22-ba0b-184fe880f381
all_states = Dict();

# ╔═╡ 36e0ad57-08bd-4ece-81ba-df180d9c476f
begin
    for (k, v) in all_states # housholding
        if v.time < time() + 86400
            pop!(all_states, k)
        end
    end
    @bind(user_id, html"""<script>
            currentScript.value = Math.random()
        </script>""")
end

# ╔═╡ 402d6e7b-fdd1-4082-a995-d78fc7d4cb69
create_initial_state() = (time = time(),
                          chasse = ChasseAuTresorEnv(),
                          tictactoe = TicTacToeEnv(),
                          qlearner = QLearner(),
                          mclearner = MCLearner(),
                          cwenv = CliffWalkingEnv());

# ╔═╡ e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
let
    states = get!(all_states, user_id, create_initial_state())
    chasse = states.chasse
    chasse.action = chasse_actions[1] == 0 ? 3 : chasse_actions[2]
    if chasse.state ≤ 5 || (isa(chasse.action, Int) && chasse.action > 2)
        _learner = if learner == "mclearner"
            if length(chasse.episode_recorder) > 0 && chasse.action == 3
                update!(states.mclearner, chasse.episode_recorder)
            end
            states.mclearner
        else
            if length(chasse.episode_recorder) > 0
                update!(states.qlearner, last(chasse.episode_recorder)...)
            end
            states.qlearner
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

# ╔═╡ e876c526-30f9-458d-abf5-e20e6aa0268e
let
    chasse_actions
    states = get!(all_states, user_id, create_initial_state())
    if learner == "mclearner"
        showQ(states.mclearner.Q)
    else
        showQ(states.qlearner.Q)
    end
end

# ╔═╡ ebebd97a-9dc2-4b39-a998-9279d52c57e5
let
    states = get!(all_states, user_id, create_initial_state())
    tictactoe = states.tictactoe
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
        if reward(tictactoe) == -1
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
    states = get!(all_states, user_id, create_initial_state())
    cwenv = states.cwenv
    a = last(cw_action)
    if a == 5
        r = 0
        cwenv.cumulative_reward = 0
        reset!(cwenv)
    elseif a == 0
        r = 0
    elseif !is_terminated(cwenv)
        act!(cwenv, a)
        r = reward(cwenv)
        cwenv.cumulative_reward += r
	else
		r = reward(cwenv)
    end;
    plot(cwenv)
    plot!(size = (700, 400))
    annotate!([(cwenv.params.ny ÷ 2, -1, "reward = $r"),
               (cwenv.params.ny ÷ 2, -.2, "cumulative reward = $(cwenv.cumulative_reward)")])
end

# ╔═╡ Cell order:
# ╠═b97724e4-d7b0-4085-b88e-eb3c5bcbe441
# ╟─b3793299-c916-40d3-bd87-31153fc3781a
# ╟─ce405f97-6d60-4ae4-b183-79e6c88d9811
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
# ╠═c16fbc62-3dcc-4097-bc82-ec26d89794ec
# ╠═31da392f-0283-4a50-adfa-ac1c14ad2ac3
# ╠═b06097e9-ef1e-4a18-839e-9e3758e5201b
# ╠═bb299fa3-925f-4f12-89a0-aa890bf56c25
# ╠═af7015c4-e7ab-4e18-bd37-ccffe4ec2928
# ╠═ce334e27-9b66-4692-becd-cfc24ff58cb1
# ╟─6464f6d3-2fcf-493f-ad21-206d9c273a16
# ╠═3a643502-7d78-4d0c-a53f-913f35306258
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
# ╟─d88c1d9b-3396-42a2-8ebd-81851f778602
# ╟─761d690d-5c73-40dd-b38c-5af67ee837c0
# ╟─37d0f305-3aea-4e22-ba0b-184fe880f381
# ╟─36e0ad57-08bd-4ece-81ba-df180d9c476f
# ╟─402d6e7b-fdd1-4082-a995-d78fc7d4cb69
