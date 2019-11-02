using Flux
using Flux.Tracker
using Distributions
using Revise



struct VariationalAutoencoder

    encoder::Flux.Chain
    decoder::Flux.Chain
    likelihood::Distributions.Distribution

    encode::Function
    decode::Function

end

function VariationalAutoencoder(encoder::Flux.Chain, decoder::Flux.Chain,
                                likelihood::Distributions.Distribution)

    encode(x) = encoder(x)
    decode(x) = decoder(x)

    return VariationalAutoencoder(encoder, decoder, likelihood, encode, decode)
end

struct VariationalEncoderLayer
    Wμ
    bμ

    Wσ
    bσ

    activation
end
Flux.@treelike VariationalEncoderLayer

function VariationalEncoderLayer(in::Integer, out::Integer, activation::Function)
    VariationalEncoderLayer(Flux.param(randn(out,in)), Flux.param(randn(out)),
                     Flux.param(randn(out,in)), Flux.param(randn(out)),
                     activation)
end

(m::VariationalEncoderLayer)(x) = [m.Wμ * x .+ m.bμ, m.activation.(m.Wσ * x .+ m.bσ)]



struct VariationalDecoderLayer
    W
    b
    activation
end
Flux.@treelike VariationalDecoderLayer

function VariationalDecoderLayer(in::Integer, out::Integer, activation::Function)
    VariationalDecoderLayer(Flux.param(randn(out,in)), Flux.param(randn(out)),
                     activation)
end

function (m::VariationalDecoderLayer)(μ, σ)
    sample = rand.(Distributions.Normal.(μ, σ))
    return m.activation.(m.W * sample .+ m.b)
end

function (m::VariationalDecoderLayer)(parameters)

    μ = parameters[1]
    σ = parameters[2]

    sample = rand.(Distributions.Normal.(μ, σ))
    return m.activation.(m.W * sample .+ m.b)
end






function calcStandardNormalKLD(μ, σ)
    return -2*log(σ) + σ^2 + μ^2 - 1
end

function calcLikelihood(dist::Distribution, x)
    return logpdf(dist, x)
end
