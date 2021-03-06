export RainbowLearner

using Flux
using Zygote
using StatsBase
using Random
using LinearAlgebra: dot

"""
    RainbowLearner(;kwargs...)

See paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `Vₘₐₓ::Float32`: the maximum value of distribution.
- `Vₘᵢₙ::Float32`: the minimum value of distribution.
- `n_actions::Int`: number of possible actions.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `target_update_freq::Int=500`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `default_priority::Float32=1.0f2.`: the default priority for newly added transitions. It must be `>= 1`.
- `n_atoms::Int=51`: the number of buckets of the value function distribution.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `rng = Random.GLOBAL_RNG`
"""
mutable struct RainbowLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    Ts,
    Tss<:Union{Int,Nothing},
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    Vₘₐₓ::Float32
    Vₘᵢₙ::Float32
    n_actions::Int
    n_atoms::Int
    support::Ts
    stack_size::Tss
    delta_z::Float32
    γ::Float32
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float32
    β_priority::Float32
    rng::R
    loss::Float32
end

Flux.functor(x::RainbowLearner) =
    (Q = x.approximator, Qₜ = x.target_approximator, S = x.support),
    y -> begin
        x = @set x.approximator = y.Q
        x = @set x.target_approximator = y.Qₜ
        x = @set x.support = y.S
        x
    end

function RainbowLearner(;
    approximator,
    target_approximator,
    loss_func,
    Vₘₐₓ,
    Vₘᵢₙ,
    n_actions,
    n_atoms = 51,
    support = collect(range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length = n_atoms)),
    stack_size = 4,
    delta_z = Float32(support[2] - support[1]),
    γ = 0.99,
    batch_size = 32,
    update_horizon = 1,
    min_replay_history = 32,
    update_freq = 1,
    target_update_freq = 500,
    update_step = 0,
    default_priority = 1.0f2,
    β_priority = 0.5f0,
    rng = Random.GLOBAL_RNG,
)
    default_priority >= 1.0f0 || error("default value must be >= 1.0f0")
    copyto!(approximator, target_approximator)  # force sync
    support = send_to_device(device(approximator), support)
    RainbowLearner(
        approximator,
        target_approximator,
        loss_func,
        Vₘₐₓ,
        Vₘᵢₙ,
        n_actions,
        n_atoms,
        support,
        stack_size,
        delta_z,
        γ,
        batch_size,
        update_horizon,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        default_priority,
        β_priority,
        rng,
        0.0f0,
    )
end

function (learner::RainbowLearner)(env)
    state = send_to_device(device(learner.approximator), get_state(env))
    state = Flux.unsqueeze(state, ndims(state) + 1)
    logits = learner.approximator(state)
    q = learner.support .* softmax(reshape(logits, :, learner.n_actions))
    vec(sum(q, dims = 1)) |> send_to_host
end

function RLBase.update!(learner::RainbowLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.γ
    β = learner.β_priority
    loss_func = learner.loss_func
    n_atoms = learner.n_atoms
    n_actions = learner.n_actions
    support = learner.support
    delta_z = learner.delta_z
    update_horizon = learner.update_horizon
    batch_size = learner.batch_size
    D = device(Q)
    states = send_to_device(D, batch.states)
    rewards = send_to_device(D, batch.rewards)
    terminals = send_to_device(D, batch.terminals)
    next_states = send_to_device(D, batch.next_states)

    actions = CartesianIndex.(batch.actions, 1:batch_size)

    target_support =
        reshape(rewards, 1, :) .+
        (reshape(support, :, 1) * reshape((γ^update_horizon) .* (1 .- terminals), 1, :))

    next_logits = Qₜ(next_states)
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
    next_q = reshape(sum(support .* next_probs, dims = 1), n_actions, :)
    if !isnothing(batch.next_legal_actions_mask)
        masked_value = fill(typemin(Float32), size(batch.next_legal_actions_mask))
        masked_value[batch.next_legal_actions_mask] .= 0
        next_q .+= send_to_device(D, masked_value)
    end
    next_prob_select = select_best_probs(next_probs, next_q)

    target_distribution = project_distribution(
        target_support,
        next_prob_select,
        support,
        delta_z,
        learner.Vₘᵢₙ,
        learner.Vₘₐₓ,
    )

    is_use_PER = !isnothing(batch.priorities)  # is use Prioritized Experience Replay
    if is_use_PER
        updated_priorities = Vector{Float32}(undef, batch_size)
        weights = 1.0f0 ./ ((batch.priorities .+ 1f-10) .^ β)
        weights ./= maximum(weights)
        weights = send_to_device(D, weights)
    end

    gs = gradient(Flux.params(Q)) do
        logits = reshape(Q(states), n_atoms, n_actions, :)
        select_logits = logits[:, actions]
        batch_losses = loss_func(select_logits, target_distribution)
        loss =
            is_use_PER ? dot(vec(weights), vec(batch_losses)) * 1//batch_size :
            mean(batch_losses)
        ignore() do
            if is_use_PER
                updated_priorities .= send_to_host(vec((batch_losses .+ 1f-10) .^ β))
            end
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)

    updated_priorities
end

@inline function select_best_probs(probs, q)
    q_argmax = argmax(q, dims = 1)
    prob_select = @inbounds probs[:, q_argmax] # !!! without @inbounds it would be really slow
    reshape(prob_select, :, length(q_argmax))
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(
        repeat(clampped_support; outer = (n_atoms, 1)),
        n_atoms,
        n_atoms,
        batch_size,
    )

    projection =
        clamp.(
            1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z,
            0,
            1,
        ) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims = 1), n_atoms, batch_size)
end
