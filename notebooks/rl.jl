commit 43d89ac6882f760526563dace3a252f8150cb994
Author: Johanni Brea <jbrea@users.noreply.github.com>
Date:   Wed Apr 6 19:23:00 2022 +0200

    up

diff --git a/notebooks/rl.jl b/notebooks/rl.jl
index 42a1bc0..40c623f 100644
--- a/notebooks/rl.jl
+++ b/notebooks/rl.jl
@@ -1,5 +1,5 @@
 ### A Pluto.jl notebook ###
-# v0.17.7
+# v0.18.0
 
 using Markdown
 using InteractiveUtils
@@ -24,8 +24,8 @@ end
 # ╔═╡ b3793299-c916-40d3-bd87-31153fc3781a
 using PlutoUI, PlutoUI.BuiltinsNotebook.HypertextLiteral; PlutoUI.TableOfContents()
 
-# ╔═╡ 1a17e9b2-a439-4521-842c-96ebe0378919
-using ReinforcementLearning
+# ╔═╡ bb299fa3-925f-4f12-89a0-aa890bf56c25
+using Flux # used to define the Qnetwork as a feedforward neural network
 
 # ╔═╡ ce405f97-6d60-4ae4-b183-79e6c88d9811
 md"# Chasse au trésor"
@@ -93,6 +93,96 @@ end
 # ╔═╡ 0537554c-76e4-4827-b92a-bf5b944685c7
 md"Now we let this agent play the game for 10^6 episodes and learn the Q-values with Monte Carlo estimation."
 
+
+# ╔═╡ 95ee4cf6-afd8-4979-b907-10d13aa3b079
+md"# Markov Decision Processes
+
+## A Side-Remark about Fixed-Point Iteration
+
+Some equations can be nicely solved by a fixed point iteration.
+
+A simple example is the Babylonian method for computing the square root.
+If we transform the equation ``x^2 = a`` to ``2x^2 = a + x^2`` and finally to ``x = \frac12\left(\frac{a}x + x\right) = f_a(x)`` we get an equation ``x = f_a(x)`` that can be solved by fixed point iteration, i.e. we start with an arbitrary ``x^{(0)} > 0`` and compute iteratively ``x^{(k)} = f_a(x^{(k-1)})`` until ``x^{(k)} \approx x^{(k-1)}``.
+"
+
+# ╔═╡ a09a0468-645d-4d24-94ef-feb4822cf2b2
+function fixed_point_iteration(f, x0)
+    old_x = copy(x0)
+    while true
+        new_x = f(old_x)
+        new_x ≈ old_x && break # break, if the values does not change much anymore
+		old_x = copy(new_x)
+    end
+    old_x
+end
+
+# ╔═╡ 59407024-c9ff-4b0d-b4fb-aa295654b5b0
+fixed_point_iteration(x -> 1/2*(2/x + x), 10.)
+
+# ╔═╡ 0567c424-e98e-46fc-8508-f53df44d5fc7
+md"We see that our fixed point iteration gives the same result as ``\sqrt2 = ``$(sqrt(2)) (up to the precision set by ≈).
+
+Not every equation has this property that it can be used for a fixed point iteration (e.g. ``x = \frac{a}x`` would not work for computing the square root), but the Bellman equation *is* of this type.
+
+## Iterative Policy Evalution
+We will now use an iterative procedure to compute the Q-values of an arbitrary policy."
+
+# ╔═╡ aca82f30-ac46-4b41-bf01-c824859567bf
+function evaluate_policy(policy, T, R)
+    nₐ, nₛ = size(T)
+    r = mean.(R)' # this is an array of (nₐ, nₛ) average reward values
+    fixed_point_iteration(Q -> r .+ [sum(Q[policy(s′), s′]*T[a, s][s′] for s′ in 1:nₛ)
+                                     for a in 1:nₐ, s in 1:nₛ],
+                          zeros(nₐ, nₛ))
+end;
+
+# ╔═╡ 9e02b30a-ef38-4495-bfcb-6bb2ab838230
+begin
+    struct DeterministicPolicy
+        # the i'th entry of this vector is the action for state i.
+        actions::Vector{Int}
+    end
+    (policy::DeterministicPolicy)(s) = policy.actions[s]
+end
+
+# ╔═╡ 269c929e-fea1-4bf9-bd68-bb52b9c965df
+md"In the last line above we made objects of type `DeterministicPolicy` callable (see e.g. [here](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects))."
+
+# ╔═╡ 4b1055ce-5ae8-4ee2-ab9b-ac68630d1deb
+policy1 = DeterministicPolicy([1, 2, 1, 2, 1, 2, 1]);
+
+# ╔═╡ 8486e9c5-0db8-4868-9898-cfb752d4b8f8
+policy1(3)
+
+# ╔═╡ eeff23de-a926-4e66-abcb-370fdd577c3c
+md"This allows us now to evaluate `policy1` on the transition dynamics T and reward probabilites R of our chase au trésor."
+
+# ╔═╡ 11141e67-421a-4e02-a638-f03b47ffb53c
+md"## Policy Iteration
+
+The idea of policy iteration is to start with a random deterministic policy ``\pi^{(0)}(s)``, compute ``Q_{\pi^{(0)}}(s, a)`` with iterative policy evaluation and then update the policy such that ``\pi^{(1)}(s) = \arg\max_aQ_{\pi^{(0)}}(s, a)``. Policy evaluation and policy update are iterated until the policy does not change anymore in successive iterations, i.e. ``\pi^{(k)}(s) = \pi^{(k-1)}(s)``, and we have found the optimal policy.
+"
+
+# ╔═╡ 5be36d3c-db37-4566-a469-d1a793d26a87
+function policy_iteration(T, R)
+    nₐ, nₛ = size(T)
+    policy = DeterministicPolicy(rand(1:nₐ, nₛ))
+    while true
+        Q = evaluate_policy(policy, T, R)
+        policy_new = DeterministicPolicy([argmax(Q[:, s]) for s in 1:nₛ])
+        policy_new.actions == policy.actions && break
+        policy = policy_new
+    end
+    policy
+end
+
+
+# ╔═╡ f9fa7e6a-f1b9-428e-a3b2-044b26b96965
+md"We see that the optimal policy is to open the right door in the first (red) room
+and then open the right door if there is no guard and the left door otherwise.
+Although there is less often a guard in the blue room, there is less reward to get there.
+"
+
 # ╔═╡ dc7c5cb0-30b2-4427-8aeb-f312a88effd1
 md"# Q-Learning
 
@@ -122,13 +212,76 @@ begin
     reset!(learner::QLearner) = learner.Q .= 0
 end;
 
+# ╔═╡ 5dd76aaa-4aca-4fa3-b85c-0578cb178560
+md"""# Deep Q-Learning
+
+Discrete state representations, where each state has its own state number (e.g. "blue room with guard" = state 3), are fine for small problems, but in many situations the number of different states is too large to be enumerated with limited memory. Furthermore, with discrete state representations it is impossible to generalize between states. E.g. we may quickly learn that it is not a good idea to open a door with a guard in front of it, independently of the room color and the position of the door. With a discrete representation it is impossible to learn such generalizations, because state 3 is as different from state 5 as it is from state 4 (without additional information). Therefore it is often desirable to use a distributed representation of the states, where the different elements of the input vector indicate the presence of absence of certain features.
+"""
+
+# ╔═╡ c16fbc62-3dcc-4097-bc82-ec26d89794ec
+function distributed_representation(s)
+    [s == 1; # 1 if in room 1
+     s ∈ (2, 3); # 1 if in room 2
+     s ∈ (4, 5); # 1 if in room 3
+     s == 6; # 1 if in treasure room
+     s == 7; # 1 if KO
+     s ∈ (3, 5)] # 1 if guard present
+end
+
+# ╔═╡ 31da392f-0283-4a50-adfa-ac1c14ad2ac3
+distributed_representation(5)
+
+# ╔═╡ b06097e9-ef1e-4a18-839e-9e3758e5201b
+begin
+    Base.@kwdef mutable struct DeepQLearner # this defines a type DeepQLearner
+        Qnetwork
+        optimizer = ADAM()
+    end
+	"""
+	    update!(l::DeepQLearner, a, s, r)
+
+	Updates an `MLClearner` object `l` for a given
+	state `s`, action `a`, reward `r` and next state `s′`.
+	"""
+    function update!(l::DeepQLearner, s, a, r, s′)
+        Qnetwork = l.Qnetwork
+        x = distributed_representation(s)
+        x′ = distributed_representation(s′)
+        Q′ = maximum(Qnetwork(x′))
+        θ = params(Qnetwork)
+        gs = gradient(θ) do
+            (r + Q′ - Qnetwork(x)[a])^2
+        end
+        Flux.update!(l.optimizer, θ, gs)
+        l
+    end
+	"""
+	    update!(l::DeepQLearner, episode)
+
+	Updates an `DeepQLearner` oject `l` for a full episode.
+	"""
+    function update!(l::DeepQLearner, episode)
+		length(episode) > 0 || return l
+        s, a, r, s′ = last(episode)
+        update!(l, s, a, r, s′)
+        l
+    end
+end;
+
+# ╔═╡ 6464f6d3-2fcf-493f-ad21-206d9c273a16
+md" The Q-values of our DeepQLearner are not the same as in the discrete representation.
+For example, all those elements that should be zero are clearly non-zero. However, for decision making all that matters are the differences between the Q-values of action left and right in each state, and these differences match rather well (except for the terminal states where no decisions are taken anyway)."
+
 # ╔═╡ aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
 md"# Learning to Play Tic-Tac-Toe
 
 In the following example we will learn to play Tic-Tac-Toe with a Monte Carlo Learner.
-The Tic-Tac-Toe environment is loaded from the [`ReinforcementLearning` package](https://juliareinforcementlearning.org/). This package also provides multiple helper functions to interact with this environment, like `state`, `is_terminated`, `legal_action_space`, `ReinforcementLearning.reward`, `ReinforcementLearning.reset!` etc.
+The Tic-Tac-Toe environment is loaded from the [`ReinforcementLearning` package](https://juliareinforcementlearning.org/). This package also provides multiple helper functions to interact with this environment, like `state`, `is_terminated`, `legal_action_space`, etc.
 "
 
+# ╔═╡ 1a17e9b2-a439-4521-842c-96ebe0378919
+import ReinforcementLearning: TicTacToeEnv, legal_action_space, current_player, RLBase
+
 # ╔═╡ ae72b2c3-fe7b-4ffb-b45c-e589529209c7
 tictactoe = TicTacToeEnv();
 
@@ -151,7 +304,7 @@ begin
     act!(env::TicTacToeEnv, a) = env(a)
     reset!(env::TicTacToeEnv) = RLBase.reset!(env)
     reward(env::TicTacToeEnv) = RLBase.reward(env)
-    state(env::TicTacToeEnv) = RLBase.state(env, Observation{Int}())
+    state(env::TicTacToeEnv) = RLBase.state(env, RLBase.Observation{Int}())
 end;
 
 # ╔═╡ b6b835d1-8f84-4148-8d5b-c7aea6b0c312
@@ -252,7 +405,7 @@ begin
                                               CartesianIndex(1, 0),   # down
                                              ]
     end
-    Base.@kwdef mutable struct CliffWalkingEnv <: AbstractEnv
+    Base.@kwdef mutable struct CliffWalkingEnv <: RLBase.AbstractEnv
         params::CliffWalkingParams = CliffWalkingParams()
 		position::CartesianIndex{2} = params.start
         cumulative_reward::Int = 0
@@ -423,14 +576,6 @@ begin
         env.reward = 0
         empty!(env.episode_recorder)
     end
-    function distributed_representation(s)
-        [s == 1; # 1 if in room 1
-         s ∈ (2, 3); # 1 if in room 2
-         s ∈ (4, 5); # 1 if in room 3
-         s == 6; # 1 if in treasure room
-         s == 7; # 1 if KO
-         s ∈ (3, 5)] # 1 if guard present
-    end
 #     chasse = ChasseAuTresorEnv()
 end;
 
@@ -452,6 +597,15 @@ let mclearner = MCLearner(na = 2, ns = 7),
     showQ(mclearner.Q)
 end
 
+# ╔═╡ 3ab57bc9-907a-4b16-ae20-1e1cf2536e38
+showQ(evaluate_policy(policy1, T, R))
+
+# ╔═╡ 5c294e67-3590-41e1-bf40-b1bcc922f57a
+optimal_policy = policy_iteration(T, R)
+
+# ╔═╡ e74a4a44-ebe5-4596-b08e-d3caeb426f1c
+showQ(evaluate_policy(optimal_policy, T, R))
+
 # ╔═╡ bd8557bc-86f2-4ccc-93e9-a6bd843e80be
 let qlearner = QLearner(na = 2, ns = 7),
     chasse = ChasseAuTresorEnv()
@@ -469,6 +623,28 @@ let qlearner = QLearner(na = 2, ns = 7),
     showQ(qlearner.Q)
 end
 
+# ╔═╡ af7015c4-e7ab-4e18-bd37-ccffe4ec2928
+dql = let deepqlearner = DeepQLearner(Qnetwork = Chain(Dense(6, 10, relu),
+                                                       Dense(10, 2))),
+    chasse = ChasseAuTresorEnv()
+    for episode in 1:10^5
+        for steps in 1:2
+            x = distributed_representation(chasse.state)
+            a = epsilon_greedy_policy(deepqlearner.Qnetwork(x))
+            act!(chasse, a)
+            update!(deepqlearner, chasse.episode_recorder)
+        end
+        reset!(chasse)
+    end
+    deepqlearner
+end;
+
+# ╔═╡ 3a643502-7d78-4d0c-a53f-913f35306258
+[argmax(dql.Qnetwork(distributed_representation(s))) for s in 1:7]
+
+# ╔═╡ ce334e27-9b66-4692-becd-cfc24ff58cb1
+showQ(hcat([dql.Qnetwork(distributed_representation(s)) for s in 1:7]...))
+
 # ╔═╡ ddd02bd9-9577-44a8-8bb6-2b1d11938121
 state(tictactoe)
 
@@ -633,7 +809,7 @@ let
     end
     cpl = split(string(current_player(tictactoe)), ".")[end][1:end-2]
     s = if is_terminated(tictactoe)
-        if ReinforcementLearning.reward(tictactoe) == -1
+        if reward(tictactoe) == -1
             "$cpl lost the game."
         else
             "The game ended in a draw."
@@ -684,10 +860,35 @@ end
 # ╠═64863a37-1c25-4d1c-9f2d-33d87b4039a3
 # ╟─0537554c-76e4-4827-b92a-bf5b944685c7
 # ╠═712c2a9e-4413-4d7a-b729-cfb219723256
+# ╟─95ee4cf6-afd8-4979-b907-10d13aa3b079
+# ╠═a09a0468-645d-4d24-94ef-feb4822cf2b2
+# ╠═59407024-c9ff-4b0d-b4fb-aa295654b5b0
+# ╟─0567c424-e98e-46fc-8508-f53df44d5fc7
+# ╠═aca82f30-ac46-4b41-bf01-c824859567bf
+# ╠═9e02b30a-ef38-4495-bfcb-6bb2ab838230
+# ╟─269c929e-fea1-4bf9-bd68-bb52b9c965df
+# ╠═4b1055ce-5ae8-4ee2-ab9b-ac68630d1deb
+# ╠═8486e9c5-0db8-4868-9898-cfb752d4b8f8
+# ╟─eeff23de-a926-4e66-abcb-370fdd577c3c
+# ╠═3ab57bc9-907a-4b16-ae20-1e1cf2536e38
+# ╟─11141e67-421a-4e02-a638-f03b47ffb53c
+# ╠═5be36d3c-db37-4566-a469-d1a793d26a87
+# ╠═5c294e67-3590-41e1-bf40-b1bcc922f57a
+# ╠═e74a4a44-ebe5-4596-b08e-d3caeb426f1c
+# ╟─f9fa7e6a-f1b9-428e-a3b2-044b26b96965
 # ╟─dc7c5cb0-30b2-4427-8aeb-f312a88effd1
 # ╠═e39227c8-8148-43cb-9351-774682b65646
 # ╟─bf600a9b-484e-4a7b-bf3b-409ebad51cd0
 # ╠═bd8557bc-86f2-4ccc-93e9-a6bd843e80be
+# ╟─5dd76aaa-4aca-4fa3-b85c-0578cb178560
+# ╠═c16fbc62-3dcc-4097-bc82-ec26d89794ec
+# ╠═31da392f-0283-4a50-adfa-ac1c14ad2ac3
+# ╠═b06097e9-ef1e-4a18-839e-9e3758e5201b
+# ╠═bb299fa3-925f-4f12-89a0-aa890bf56c25
+# ╠═af7015c4-e7ab-4e18-bd37-ccffe4ec2928
+# ╠═ce334e27-9b66-4692-becd-cfc24ff58cb1
+# ╟─6464f6d3-2fcf-493f-ad21-206d9c273a16
+# ╠═3a643502-7d78-4d0c-a53f-913f35306258
 # ╟─aff5b0d2-c5e5-4fda-a93c-54a8bca5187f
 # ╠═1a17e9b2-a439-4521-842c-96ebe0378919
 # ╠═ae72b2c3-fe7b-4ffb-b45c-e589529209c7
