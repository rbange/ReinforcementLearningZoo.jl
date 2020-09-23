export NAFPolicy, NAFNetwork

using Random
using Flux
using Flux.Losses: mse
using LinearAlgebra: Diagonal, tril

# Define NAF Actor
struct NAFNetwork
    X::Flux.Chain
    MU::Any
    V::Any
    L::Any
    function NAFNetwork(X, MU, V, L)
        @assert length(size(L.W, 1))^2 == length(size(MU.W, 1)) "L size is not quadratic to action size"
        new(X, MU, V, L)
    end
end
Flux.@functor NAFNetwork
(n::NAFNetwork)(s) = (x = n.X(s); (n.MU(x), n.V(x), n.L(x)))

get_μ(n::NAFNetwork, state) = (x = n.X(state); n.MU(x))
get_v(n::NAFNetwork, state) = (x = n.X(state); n.V(x))

function get_advantage(l, action)
    temp = vec_to_square_matrix(l) # create square matrix from l
    temp2 = tril(temp, -1) + Diagonal(exp.(temp)) # transform to positive definite lower triangular matrix and exp diagonal
    p = temp2 * temp2'
    #advantage term
    -(action' * p * action) / 2
end
vec_to_square_matrix(v::Vector) = (dim = Int(sqrt(length(v))); reshape(v, dim, dim))
vec_to_square_matrix(v::Vector, dim::Int) = reshape(v, dim, dim)

mutable struct NAFPolicy{Q<:NeuralNetworkApproximator,P,R<:AbstractRNG} <: AbstractPolicy
    q::Q
    q_target::Q
    γ::Float32
    ρ::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    act_limit::Float64
    act_noise::Float64
    step::Int
    rng::R
    # for logging
    loss::Float32
end

"""
    NAFPolicy(;kwargs...)

See paper: [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)

# Keyword arguments

- `q`,
- `q_target`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `act_limit = 1.0`,
- `act_noise = 0.1`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function NAFPolicy(;
    q,
    q_target,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    α = 0.2f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_limit = 1.0,
    act_noise = 0.1,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(q, q_target)  # force sync
    NAFPolicy(
        q,
        q_target,
        γ,
        ρ,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        act_limit,
        act_noise,
        step,
        rng,
        0.0f0, # loss
    )
end

# TODO: handle Training/Testing mode
function (p::NAFPolicy)(env)
    p.step += 1

    if p.step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.q)
        s = get_state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = get_μ(p.q.model, send_to_device(D, s)) |> vec |> send_to_host
        clamp(action[] + randn(p.rng) * p.act_noise, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(p::NAFPolicy, traj::CircularCompactSARTSATrajectory)
    length(traj[:terminal]) > p.update_after || return
    p.step % p.update_every == 0 || return

    inds = rand(p.rng, 1:(length(traj[:terminal])-1), p.batch_size)
    s = select_last_dim(traj[:state], inds)
    a = select_last_dim(traj[:action], inds)
    r = select_last_dim(traj[:reward], inds)
    t = select_last_dim(traj[:terminal], inds)
    s′ = select_last_dim(traj[:next_state], inds)

    γ, ρ = p.γ, p.ρ

    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    v′ = get_v(p.q_target.model, s′)
    y = r .+ γ .* (1 .- t) .* vec(v′)

    # Train Q Networks
    a = Flux.unsqueeze(a, 1)

    q_grad = gradient(Flux.params(p.q)) do
        μ, v, l = p.q(s)
        actions = a - μ
        matrices = [l[:, i] for i = 1:p.batch_size]
        actions2 = [actions[:, i] for i = 1:p.batch_size]
        adv = hcat(get_advantage.(matrices, actions2)...)
        q = adv + v
        loss = Flux.Losses.mse(vec(q), y)
        ignore() do
            p.loss = loss
        end
        loss
    end
    update!(p.q, q_grad)

    # polyak averaging
    for (dest, src) in zip(Flux.params(p.q_target), Flux.params(p.q))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end
