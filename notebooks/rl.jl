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
@bind action Radio(["open left door", "open right door", "reset episode", "reset game"],
                   default = "reset game")

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
        episode_recorder = Tuple{Int, Int, Float64}[]
        action::String = ""
    end
    function act!(game_state::GameState, a)
        length(game_state.episode_recorder) == 2 && return
        r = reward(game_state.state, a)
        game_state.reward = r
        push!(game_state.episode_recorder, (game_state.state, a, r))
        game_state.state = act(game_state.state, a)
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
md"## Learning Q-Values with Monte Carlo Estimation"

# ╔═╡ 7c756af7-08a0-4878-8f53-9eaa79c16c1a
begin
    Base.@kwdef struct MCLearner
        N = zeros(Int, size(T)...)
        Q = zeros(size(T)...)
    end
    function update!(l::MCLearner, a, s, r)
        N, Q = l.N, l.Q
        n = N[a, s]
        Q[a, s] = 1/(n+1) * (n * Q[a, s] + r)
        N[a, s] += 1
        l
    end
    function update!(l::MCLearner, episode)
        length(episode) == 2 || return l
		G = 0
        for (s, a, r) in reverse(episode)
			G += r
            update!(l, a, s, G)
        end
        l
    end
    mclearner = MCLearner()
end;

# ╔═╡ e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
let
    do_it
    if game_state.action == "reset game" || game_state.action == "reset episode"
        update!(mclearner, game_state.episode_recorder)
        reset!(game_state)
    elseif game_state.action == "open left door"
        act!(game_state, 1)
    elseif game_state.action == "open right door"
        act!(game_state, 2)
    end
    if game_state.action == "reset game"
        mclearner.Q .= 0
        mclearner.N .= 0
    end
    if game_state.state == 1
        room = 1
        guard = false
        gold = false
    elseif game_state.state == 2
        room = 2
        guard = false
        gold = false
    elseif game_state.state == 3
        room = 2
        guard = true
        guard_door = 0
        gold = false
    elseif game_state.state == 4
        room = 3
        guard = false
        gold = false
    elseif game_state.state == 5
        room = 3
        guard = true
        guard_door = 1
        gold = false
    elseif game_state.state == 6
        room = 4
        guard = false
        gold = true
    elseif game_state.state == 7
        room = 5
        guard = false
        gold = false
    end
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
    rs = length(game_state.episode_recorder) == 0 ? [0] : last.(game_state.episode_recorder)
    annotate!([(.5, .9, "reward = $(game_state.reward)", :white),
               (.5, .8, "cumulative reward = $(join(rs, " + ")) = $(sum(rs))", :white)
              ])
end

# ╔═╡ e876c526-30f9-458d-abf5-e20e6aa0268e
let do_it
    show(mclearner)
end

# ╔═╡ fe5ae6b8-bc5e-453a-96fd-f3e7dab135e5
begin
    function Base.show(io::IO, mcl::MCLearner)
        df = DataFrame(mcl.Q, ["red room", "blue without guard", "blue with guard", "green without guard", "green with guard", "treasure room", "K.O."])
        df.action = ["left", "right"]
        df
    end
end;

# ╔═╡ 6dced47e-cc0f-4ae5-bdd0-a551c6d5a5b3
md"## An Epsilon-Greedy Agent"

# ╔═╡ 64863a37-1c25-4d1c-9f2d-33d87b4039a3
function policy(state, Q)
    if rand() < .1 # this happens with probability 0.1
        rand(1:2)  # random exploration
    else
        q = Q[:, state]
        rand(findall(==(maximum(q)), q)) # greedy exploitation with ties broken
    end
end

# ╔═╡ 712c2a9e-4413-4d7a-b729-cfb219723256
let mclearner = MCLearner(),
    game_state = GameState()
    for episode in 1:10^5
        for steps in 1:2
            s = game_state.state
            a = policy(s, mclearner.Q)
            act!(game_state, a)
        end
        update!(mclearner, game_state.episode_recorder)
        reset!(game_state)
    end
    show(mclearner)
end

# ╔═╡ 95ee4cf6-afd8-4979-b907-10d13aa3b079
md"# Markov Decision Processes"

# ╔═╡ aca82f30-ac46-4b41-bf01-c824859567bf
function evaluate_policy(policy, T, R)
    nₐ, nₛ = size(T)
    Q = zeros(nₐ, nₛ)
    r = mean.(R)'
    while true
        Qnew = r .+ [T[a, s]' * [Q[policy(s′), s′] for s′ in 1:nₛ]
                     for a in 1:nₐ, s in 1:nₛ]
        Qnew ≈ Q && break
        Q = Qnew
    end
    Q
end;

# ╔═╡ 9e02b30a-ef38-4495-bfcb-6bb2ab838230
begin
    struct DeterministicPolicy
        actions::Vector{Int}
    end
    (policy::DeterministicPolicy)(s) = policy.actions[s]
end

# ╔═╡ 3ab57bc9-907a-4b16-ae20-1e1cf2536e38
evaluate_policy(DeterministicPolicy([1, 2, 1, 2, 1, 2, 1]), T, R)

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
evaluate_policy(optimal_policy, T, R)

# ╔═╡ 412d8fcb-8f98-43b6-9235-a4c228317427
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─ce405f97-6d60-4ae4-b183-79e6c88d9811
# ╟─e98b3e4f-d17e-4fdd-af1c-a8744ce7ecc3
# ╟─12697628-d520-4c3a-9e33-f7c8e17f0dea
# ╟─86a4f1ef-c81c-4888-95ed-15269d38fbea
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
# ╠═aca82f30-ac46-4b41-bf01-c824859567bf
# ╠═9e02b30a-ef38-4495-bfcb-6bb2ab838230
# ╠═3ab57bc9-907a-4b16-ae20-1e1cf2536e38
# ╠═5be36d3c-db37-4566-a469-d1a793d26a87
# ╠═5c294e67-3590-41e1-bf40-b1bcc922f57a
# ╠═e74a4a44-ebe5-4596-b08e-d3caeb426f1c
# ╠═1a17e9b2-a439-4521-842c-96ebe0378919
# ╟─8bd459cb-20bb-483e-a849-e18caae3beef
# ╟─b97724e4-d7b0-4085-b88e-eb3c5bcbe441
# ╟─412d8fcb-8f98-43b6-9235-a4c228317427
