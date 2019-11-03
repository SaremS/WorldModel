using Flux
using Flux.Tracker
using Distributions
using Revise



struct VariationalAutoencoder
    encoder::Flux.Chain
    decoder::Flux.Chain
    distribution
end

function VariationalAutoencoder(encoder::Flux.Chain, decoder::Flux.Chain)
    return VariationalAutoencoder(encoder, decoder, Bernoulli)
end

struct VariationalEncoderLayer
    Wμ
    bμ
    Wσ
    bσ

    activation
    distribution
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




function vae_encode(vae::VariationalAutoencoder, x)
    return vae.encoder(x)
end

function vae_decode(vae::VariationalAutoencoder, x)
    return vae.decoder(x)
end

function (m::VariationalAutoencoder)(x)
    encoding(x) = vae_encode(m, x)
    decoding(x) = vae_decode(m, x)

    encoded = [encoding([x[i]]) for i in 1:size(x)[1]]
    return Tracker.collect(decoding.(encoded))
end




function calcStandardNormalKLD(μ, σ)
    return -2*log(σ) + σ^2 + μ^2 - 1
end

function calcStandardNormalKLD(parameters)
    μ = parameters[1]
    σ = parameters[2]
    return calcStandardNormalKLD(μ, σ)
end

function calcLogLikelihood(dist::Distribution, x)
    return logpdf(dist, x)
end


function vae_logpdf(vae::VariationalAutoencoder, x)
    
end
