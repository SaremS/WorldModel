using Flux
using Flux.Tracker
using Distributions
using Revise



struct VariationalAutoencoder
    encoder::Flux.Chain
    decoder::Flux.Chain
    distribution
end
@Flux.treelike VariationalAutoencoder

function VariationalAutoencoder(encoder::Flux.Chain, decoder::Flux.Chain)
    return VariationalAutoencoder(encoder, decoder, Bernoulli)
end

struct VariationalEncoderLayer
    Wμ
    Wσ

    activation
end
Flux.@treelike VariationalEncoderLayer

function VariationalEncoderLayer(in::Integer, out::Integer, activation::Function)
    VariationalEncoderLayer(Flux.param(randn(out,in)),
                     Flux.param(randn(out,in)),
                     activation)
end

(m::VariationalEncoderLayer)(x) = [m.Wμ * x, m.activation.(m.Wσ * x)]



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
    return sum(- log.(σ .^2) .+ σ.^2 .+ μ.^2 .- 1)
end

function calcStandardNormalKLD(parameters)
    μ = parameters[1]
    σ = parameters[2]
    return calcStandardNormalKLD(μ, σ)
end

function calcLogLikelihood(dist::Distribution, x)
    return logpdf(dist, x)
end


function calcVaeLogpdf(vae::VariationalAutoencoder, x)

    dist = vae.distribution

    sample = [vae(x[i])[1][1] for i in 1:size(x)[1]]
    println(sample)
    println(dist.(sample))

    return -mean(logpdf.(dist.(sample), x))

end


function calcVariationalLoss(vae::VariationalAutoencoder, x, n_samples = 100)

    inner_sample = (vae.encoder.(x))

    kld = mean(calcStandardNormalKLD.(inner_sample))



    outer_sample = [vae.decoder(inner_sample[i])[1][1] for i in 1:size(x)[1]]
    dist = vae.distribution
    loglike = Tracker.collect(-mean(logpdf.(dist.(outer_sample), x)))[1]

    return loglike + kld
end


encoder = Chain(Dense(1, 20), VariationalEncoderLayer(20,1, x->relu(x)+1e-6))
decoder = Chain(VariationalDecoderLayer(1,20,relu), Dense(20,1, σ))

test = VariationalAutoencoder(encoder, decoder)


calcVariationalLoss(test, [[1.], [0.]])

paramets = Flux.params(test)

opt = ADAM()
#Flux.train!(loss, paramets, zip([[1.], [0.]]), opt)
for i in 1:1000
    loss(x) = calcVariationalLoss(test, x)
    print(loss([[1.],[0.]]))
    gradients = Tracker.gradient(()->loss([[1.], [0.]]), paramets)


    for p in paramets
        Tracker.update!(opt, p, gradients[p])
    end
end
mean([test([1.]).data[1][1] for i in 1:100000])
